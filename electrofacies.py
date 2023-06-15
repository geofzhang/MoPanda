import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import (
    MiniBatchKMeans,
    DBSCAN,
    AffinityPropagation,
    OPTICS,
    SpectralClustering,
)


def electrofacies(
    logs,
    formations,
    curves,
    log_scale=None,
    n_components=0.85,
    curve_names=None,
    clustering_methods=None,
    clustering_params=None,
    depth_constrain_weight=0.5,
):
    """
    Electrofacies function to group intervals by rock type. Also
    referred to as heterogenous rock analysis.

    Parameters
    ----------
    logs : list of :class:`petropy.Log` objects
        List of Log objects
    formations : list of formation names
        List of str containing formation names which should be
        previously loaded into Log objects
    curves : list of curve names
        List of strings containing curve names as inputs in the
        electrofacies calculations
    n_clusters : int
        Number of clusters to group intervals. Number of electrofacies.
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
        'kmeans', 'dbscan', 'affinity', 'optics', 'spectral'.
        If None, all clustering methods will be used.
    clustering_params : dict or None (default None)
        Additional parameters for clustering algorithms. The keys of the dictionary
        should match the clustering method names, and the values should be dictionaries
        of parameter names and values for each clustering algorithm.
    depth_constrain_weight : float (default 0.5)
        Weight factor for incorporating the depth constraint. Higher values
        assign more importance to proximity in depth for clustering.

    """

    if not log_scale:
        log_scale = []

    if not clustering_methods:
        clustering_methods = [
            'kmeans',
            'dbscan',
            'affinity',
            'optics',
            'spectral',
        ]

    if not curve_names:
        curve_names = ['FACIES_' + method.upper() for method in clustering_methods]

    if not clustering_params:
        clustering_params = {}

    df = pd.DataFrame()

    for log in logs:

        if log.well['UWI'] is None:
            raise ValueError('UWI required for log identification.')

        log_df = log.df().reset_index()
        log_df['UWI'] = log.well['UWI'].value
        log_df['DEPTH_INDEX'] = np.arange(0, len(log[0]))

        for formation in formations:
            top = log.tops[formation]
            bottom = log.next_formation_depth(formation)
            depth_index = np.intersect1d(
                np.where(log[0] >= top)[0], np.where(log[0] < bottom)[0]
            )
            df = df.append(log_df.iloc[depth_index])

    for s in log_scale:
        df[s] = np.log(df[s])

    not_null_rows = pd.notnull(df[curves]).any(axis=1)

    X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])

    pc = PCA(n_components=n_components).fit(X)

    components = pd.DataFrame(data=pc.transform(X), index=df[not_null_rows].index)
    minibatch_input = components.to_numpy()
    components.columns = [f'PC{i}' for i in range(1, pc.n_components_ + 1)]
    components['UWI'] = df.loc[not_null_rows, 'UWI']
    components['DEPTH_INDEX'] = df.loc[not_null_rows, 'DEPTH_INDEX']

    size = len(components) // 20
    size = 10000 if size > 10000 else 100 if size < 100 else size

    for method, curve_name in zip(clustering_methods, curve_names):
        clustering_param = clustering_params.get(method, {})
        if method == 'kmeans':
            model = MiniBatchKMeans(batch_size=size, **clustering_param)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5, **clustering_param)
        elif method == 'affinity':
            model = AffinityPropagation(**clustering_param)
        elif method == 'optics':
            model = OPTICS(min_samples=5, **clustering_param)
        elif method == 'spectral':
            model = SpectralClustering(**clustering_param)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        distance_matrix = pairwise_distances(minibatch_input, metric='euclidean')
        depth_constrain = np.exp(-depth_constrain_weight * np.abs(np.subtract.outer(components['DEPTH_INDEX'], components['DEPTH_INDEX'])))
        weighted_distance_matrix = distance_matrix * depth_constrain

        if method == 'kmeans':
            labels = model.fit_predict(weighted_distance_matrix) + 1
        else:
            labels = model.fit_predict(weighted_distance_matrix)

        df.loc[not_null_rows, curve_name] = labels

    for log in logs:

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
            data[depth_index] = np.copy(df.loc[df.UWI == uwi, curve_name])
            log.add_curve(curve_name, np.copy(data), descr='Electrofacies')

    return logs
