import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    MiniBatchKMeans,
    DBSCAN,
    AffinityPropagation,
    OPTICS,
)
import PySimpleGUI as sg
import skfuzzy as fuzz


def select_curves(log):
    """
    Prompt the user to select curves for electrofacies analysis.

    Parameters
    ----------
    log : :class:`petropy.Log` object
        Log object containing available curves.

    Returns
    -------
    list of strings
        Selected curve mnemonics.

    """

    # Get curve names and descriptions
    curve_names = [curve.mnemonic for curve in log.curves]
    curve_descs = [curve.descr for curve in log.curves]

    # GUI layout for curve selection
    layout = [
        [sg.Text("Available curves:")],
        [sg.Listbox(list(zip(curve_names, curve_descs)), size=(60, 6), key="-CURVE_LIST-", select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE)],
        [sg.Button("Select", key="-SELECT-")]
    ]

    window = sg.Window("Curve Selection", layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        if event == "-SELECT-":
            selected_indices = values["-CURVE_LIST-"]
            selected_curves = [idx for idx, _ in selected_indices]  # Extract curve indices
            break

    window.close()

    return selected_curves


def electrofacies(
    logs,
    formations,
    curves=None,
    log_scale=None,
    n_components=0.85,
    curve_names=None,
    clustering_methods=None,
    clustering_params=None,
    depth_constrain_weight=0.5,
    cluster_range=None,
):
    """
    Electrofacies function to group intervals by rock type. Also
    referred to as heterogeneous rock analysis.

    Parameters
    ----------
    logs : list of :class:`petropy.Log` objects
        List of Log objects
    formations : list of formation names
        List of str containing formation names which should be
        previously loaded into Log objects
    curves : list of curve names or None (default None)
        List of strings containing curve names as inputs in the
        electrofacies calculations. If None, prompts user to choose curves.
    log_scale : list of curve names
        List of string containing curve names which are preprocessed
        on a log scale. For example, deep resistivity separates better
        on a log scale, and is graphed logarithmically when viewing
        data in a log viewer.
    n_components : int, float, None or string (default 0.85)
        Number of principal components to keep. If value is less than
        one, the number of principal components will be the number
        required to exceed the explained variance.
    curve_names : list of strings or None (default None)
        List of names for the output electrofacies curves for each clustering
        algorithm used. If None, default names will be assigned.
    clustering_methods : list of strings or None (default None)
        List of clustering methods to be used. Possible values:
        'kmeans', 'dbscan', 'affinity', 'optics', 'fuzzy'.
        If None, all clustering methods will be used.
    clustering_params : dict or None (default None)
        Additional parameters for clustering algorithms. The keys of the dictionary
        should match the clustering method names, and the values should be dictionaries
        of parameter names and values for each clustering algorithm.
    depth_constrain_weight : float (default 0.5)
        Weight factor for incorporating the depth constraint. Higher values
        assign more importance to proximity in depth for clustering.
    cluster_range : tuple or None (default None)
        Range of cluster numbers to evaluate for Fuzzy C-means clustering.
        If None, the range will be set to (2, 10).

    Returns
    -------
    list of :class:`petropy.Log` objects
        List of Log objects with added electrofacies curves.

    """

    if not log_scale:
        log_scale = []

    if not curves:
        layout = [
            [sg.Text("No curve names provided. Do you want to load default triple combo curves?")],
            [sg.Button("Yes", key="-DEFAULT_CURVES-"), sg.Button("No", key="-MANUAL_CURVES-")]
        ]
        window = sg.Window("Curve Selection", layout)
        event, _ = window.read()
        window.close()

        if event == "-DEFAULT_CURVES-":
            curves = ['SGR_N', 'SP_N', 'RESDEEP_N', 'NPHI_N', 'RHOB_N', 'PE_N']
        elif event == "-MANUAL_CURVES-":
            curves = select_curves(logs[0])

    if not clustering_methods:
        clustering_methods = [
            'kmeans',
            'dbscan',
            'affinity',
            'optics',
            'fuzzy',
        ]

    if not curve_names:
        curve_names = ['FACIES_' + method.upper() for method in clustering_methods]

    if not clustering_params:
        clustering_params = {}

    dfs = []

    for log in logs:

        if log.well['UWI'] is None:
            raise ValueError('UWI required for log identification.')

        df = pd.DataFrame()
        log_df = log.df().reset_index()

        log_df['UWI'] = log.well['UWI'].value
        log_df['DEPTH_INDEX'] = np.arange(0, len(log[0]))

        if not formations:
            formations = list(log.tops.keys())

        for formation in formations:
            top = log.tops[formation]
            bottom = log.formation_bottom_depth(formation)
            depth_index = np.intersect1d(
                np.where(log[0] >= top)[0], np.where(log[0] < bottom)[0]
            )
            log_df_subset = log_df.iloc[depth_index]
            log_df_subset = log_df_subset.dropna()
            df = pd.concat([df, log_df_subset], ignore_index=True)

        dfs.append(df)

    for df in dfs:

        for s in log_scale:
            df[s] = np.log(df[s])

        not_null_rows = pd.notnull(df[curves]).any(axis=1)

        X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])

        pc = PCA(n_components=n_components).fit(X)

        components = pd.DataFrame(data=pc.transform(X), index=df[not_null_rows].index)
        clustering_input = components.to_numpy()
        components.columns = [f'PC{i}' for i in range(1, pc.n_components_ + 1)]
        components['UWI'] = df.loc[not_null_rows, 'UWI']
        components['DEPTH_INDEX'] = df.loc[not_null_rows, 'DEPTH_INDEX']

        size = len(components) // 20
        size = 1024 if size > 1000 else 100 if size < 100 else size

        for method, curve_name in zip(clustering_methods, curve_names):
            clustering_param = clustering_params.get(method, {})
            model = None  # Initialize model variable

            distance_matrix = np.sqrt(np.sum((clustering_input[:, np.newaxis] - clustering_input) ** 2, axis=2))
            depth_indices = components['DEPTH_INDEX'].to_numpy()
            depth_constrain = np.exp(-depth_constrain_weight * np.abs(np.subtract.outer(depth_indices, depth_indices)))
            weighted_distance_matrix = distance_matrix * depth_constrain

            if method == 'kmeans':
                if 'n_clusters' not in clustering_param:
                    cluster_range = cluster_range or (2, 10)
                    clustering_param['n_clusters'] = find_optimal_cluster_number_kmeans(X, cluster_range)
                model = MiniBatchKMeans(batch_size=size, **clustering_param)
            elif method == 'dbscan':
                model = DBSCAN(**clustering_param)
            elif method == 'affinity':
                model = AffinityPropagation(**clustering_param)
            elif method == 'optics':
                model = OPTICS(**clustering_param)
            elif method == 'fuzzy':
                if 'n_clusters' not in clustering_param:
                    cluster_range = cluster_range or (2, 10)
                print(f'Clustering using {method} method...')
                best_n_clusters = find_optimal_cluster_number_fuzzy(weighted_distance_matrix, cluster_range)
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(weighted_distance_matrix.T, best_n_clusters, 2, error=0.005, maxiter=1000)
                membership_scores = u.T
                labels = np.argmax(membership_scores, axis=1) + 1
                for i, cluster in enumerate(range(best_n_clusters)):
                    membership_curve = f'{curve_name}_MEMBER_{cluster + 1}'
                    df.loc[not_null_rows, membership_curve] = membership_scores[:, i]
                    curve_names.append(membership_curve)
            else:
                raise ValueError(f"Unknown clustering method: {method}")

            if model is not None:
                print(f'Clustering using {method} method...')
                if method == 'kmeans':
                    labels = model.fit_predict(weighted_distance_matrix) + 1
                else:
                    labels = model.fit_predict(weighted_distance_matrix)

            df.loc[not_null_rows, curve_name] = labels
            print(f'Done!')

        for log, df in zip(logs, dfs):

            uwi = log.well['UWI'].value

            for v, vector in enumerate(pc.components_):
                v += 1
                pc_curve = f'PC{v}'

                if pc_curve in log.keys():
                    data = log[pc_curve]
                else:
                    data = np.empty(len(log[0]))
                    data[:] = np.nan

                depth_index = components.loc[components.UWI == uwi, 'DEPTH_INDEX']
                data[depth_index] = np.copy(components.loc[components.UWI == uwi, pc_curve])
                log.add_curve(
                    pc_curve,
                    np.copy(data),
                    descr=f'Principal Component {v} from electrofacies',
                )

            for curve_name in curve_names:
                if curve_name in log.keys():
                    data = log[curve_name]
                else:
                    data = np.empty(len(log[0]))
                    data[:] = np.nan

                depth_index = df.loc[df.UWI == uwi, 'DEPTH_INDEX']
                data[depth_index] = df.loc[df.UWI == uwi, curve_name]
                log.add_curve(
                    curve_name,
                    np.copy(data),
                    descr='Electrofacies',
                )

    return logs

def find_optimal_cluster_number_kmeans(X, cluster_range):
    """
    Find the optimal number of clusters for K-means clustering using the Elbow Method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    cluster_range : tuple
        Range of cluster numbers to evaluate.

    Returns
    -------
    int
        The optimal number of clusters.

    """
    scores = []
    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        model = MiniBatchKMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    best_n_clusters = np.argmax(scores) + cluster_range[0]
    print(f'The optimal cluster number is {best_n_clusters}.')
    return best_n_clusters
def find_optimal_cluster_number_fuzzy(X, cluster_range):
    """
    Find the optimal number of clusters using Fuzzy C-means clustering.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    cluster_range : tuple
        Range of cluster numbers to evaluate.

    Returns
    -------
    int
        The optimal number of clusters.

    """

    best_n_clusters = cluster_range[0]
    best_score = -np.inf

    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, 2, error=0.005, maxiter=1000)
        score = fpc.max()
        print(n_clusters, score)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    print(f'The optimal cluster number is {best_n_clusters}, with a FPC score of {best_score}.')
    return best_n_clusters

