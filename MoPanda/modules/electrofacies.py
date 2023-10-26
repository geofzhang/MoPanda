import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import PySimpleGUI as sg
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import (
    MiniBatchKMeans,
    DBSCAN,
    AffinityPropagation,
    OPTICS,
    AgglomerativeClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from modules.data_analysis import plot_pca_subplots, fill_null
from modules.utils import ColorCoding


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
        [sg.Listbox(list(zip(curve_names, curve_descs)), size=(60, 6), key="-CURVE_LIST-",
                    select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE)],
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
        cluster_range=None,
        template=None,
        template_xml_path=None,
        lithology_color_coding=None,
        masking=None
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
            [sg.Text("No log names provided. Do you want to load default logs for electrofacies classification?")],
            [sg.Button("Yes", key="-DEFAULT_CURVES-"), sg.Button("No", key="-MANUAL_CURVES-")]
        ]
        window = sg.Window("Logs for Electrofacies Classification", layout)
        event, _ = window.read()
        window.close()

        if event == "-DEFAULT_CURVES-":
            curves = ['CAL_N', 'SGR_N', 'DTC_N', 'PHIE', 'RESDEEP_N', 'NPHI_N', 'DPHI_N',
                      'RHOB_N', 'PE_N']
        elif event == "-MANUAL_CURVES-":
            curves = select_curves(logs[0])

    if not clustering_methods:
        clustering_methods = [
            'kmeans',
            'dbscan',
            'affinity',
            'agglom',
            'fuzzy',
        ]

    if not curve_names:
        curve_names = ['FACIES_' + method.upper() for method in clustering_methods]

    if not clustering_params:
        clustering_params = {}

    default_templates_paths = {
        'raw': 'default_raw_template.xml',
        'full': 'default_full_template.xml',
        'lithofacies': 'default_lithofacies_template.xml',
        'electrofacies': 'default_electrofacies_template.xml',
        'salinity': 'default_salinity_template.xml',
        'scoping_simple': 'default_scoping_simple_template.xml',
        'scoping': 'default_scoping_template.xml',
        'permeability': 'default_permeability_template.xml'
    }

    file_dir = os.path.dirname(__file__)
    if template_xml_path:
        template_xml_path = template_xml_path
    elif template_xml_path is None and template is None:
        print('No template or template path assigned, loading default raw template.')
        template_xml_path = os.path.join(file_dir, '../data/template',
                                         'default_raw_template.xml')
    else:
        if template in default_templates_paths:
            file_name = default_templates_paths[template]
        else:
            print('template_defaults paramter must be in:')
            for key in default_templates_paths:
                print(key)
            raise ValueError("%s is not valid template_defaults \
                             parameter" % template)
        template_xml_path = os.path.join(file_dir, '../data/template',
                                         file_name)
    output_template = []
    dfs = []

    for log in logs:

        if log.well['UWI'] is None:
            raise ValueError('UWI required for log identification.')

        df = pd.DataFrame()
        log_df = log.df()
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

            df = pd.concat([df, log_df_subset], ignore_index=True)
        dfs.append(df)

    for df in dfs:

        for s in log_scale:
            if s in df.columns:
                df[s] = np.log10(df[s]+0.1)

        # Cleaning and filling null data within depth range
        df = fill_null(df)

        # Remove rows with null at the beginning or end that can't be interpolated
        not_null_rows = pd.notnull(df[curves]).all(axis=1)

        X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])

        pc = PCA(n_components=n_components).fit(X)

        pc_full = PCA(n_components=min(len(curves), len(df.index))).fit(X)
        pc_full_variance = pc_full.explained_variance_ratio_

        pca_loadings = pd.DataFrame((pc.components_.T * np.sqrt(pc.explained_variance_)).T, columns=curves)

        components = pd.DataFrame(data=pc.transform(X), index=df[not_null_rows].index)
        clustering_input = components.to_numpy()
        components.columns = [f'PC{i}' for i in range(1, pc.n_components_ + 1)]
        components['UWI'] = df.loc[not_null_rows, 'UWI']
        components['DEPTH_INDEX'] = df.loc[not_null_rows, 'DEPTH_INDEX']

        size = len(components) // 20
        size = 1024 if size > 1000 else 100 if size < 100 else size

        # Initialize dictionaries to store cluster labels and averages for each method
        cluster_labels = {}
        cluster_averages = {}
        cluster_membership_scores = {}

        for method, curve_name in zip(clustering_methods, curve_names):
            clustering_param = clustering_params.get(method, {})
            model = None  # Initialize model variable

            if method == 'kmeans':
                if 'n_clusters' not in clustering_param:
                    cluster_range = cluster_range or (2, 10)
                    clustering_param['n_clusters'] = find_optimal_cluster_number_kmeans(clustering_input, cluster_range)
                model = MiniBatchKMeans(batch_size=size, **clustering_param)
            elif method == 'dbscan':
                model = DBSCAN(**clustering_param)
            elif method == 'affinity':
                model = AffinityPropagation(**clustering_param)
            elif method == 'optics':
                model = OPTICS(**clustering_param)
            elif method == 'agglom':
                model = AgglomerativeClustering(**clustering_param)
            elif method == 'fuzzy':
                print(f'Clustering using {method} method...')
                if 'n_clusters' not in clustering_param:
                    cluster_range = cluster_range or (2, 10)
                    n_clusters = find_optimal_cluster_number_fuzzy(clustering_input, cluster_range)
                else:
                    n_clusters = clustering_param['n_clusters']
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(clustering_input.T, n_clusters, 2,
                                                                 error=0.005, maxiter=1000)
                membership_scores = u.T
                labels = np.argmax(membership_scores, axis=1) + 1

                membership_curve_names = []
                for i, cluster in enumerate(range(n_clusters)):
                    membership_curve = f'{curve_name}_MEMBER_{cluster + 1}'
                    df.loc[not_null_rows, membership_curve] = membership_scores[:, i]
                    curve_names.append(membership_curve)
                    membership_curve_names.append(membership_curve)
                output_template = parsing_membership_track(template_xml_path, membership_curve_names)
                cluster_labels[method] = labels
                cluster_membership_scores[method] = membership_scores
            else:
                raise ValueError(f"Unknown clustering method: {method}")

            if model is not None:
                print(f'Clustering using {method} method...')
                if method == 'kmeans':
                    labels = model.fit_predict(clustering_input) + 1
                    num_top_logs = 10

                    # plot_pca_variance(pc_full_variance)
                    # plot_pc_crossplot(components, pca_loadings, labels, num_top_logs)
                    plot_pca_subplots(components, pca_loadings, labels, num_top_logs, pc_full_variance)

                    df.loc[not_null_rows, 'FACIES_KMEANS'] = labels  # Store labels in 'FACIES_KMEANS' column

                else:
                    labels = model.fit_predict(clustering_input) + 1
                cluster_labels[method] = labels

                cluster_averages[method] = df.loc[not_null_rows, curves].groupby(
                    labels).mean()  # Use actual curves for averages

            print(f'{len(np.unique(labels))} electrofacies assigned.')
            df.loc[not_null_rows, curve_name] = labels
            print(f'Done!')

        # Store average log responses in a table
        table_data = []
        columns = ['Method', 'Cluster'] + curves
        for method, cluster_avg in cluster_averages.items():
            if method != 'fuzzy':
                labels = cluster_labels[method]
                method_data = [[method, label] + cluster_avg.loc[label].values.tolist() for label in
                               sorted(pd.unique(labels))]
                table_data.extend(method_data)

        # Add membership scores for fuzzy clustering
        for method, membership_scores in cluster_membership_scores.items():
            labels = cluster_labels[method]
            membership_data = [[method, label] + membership_scores[i].tolist() for i, label in
                               enumerate(sorted(pd.unique(labels)))]
            table_data.extend(membership_data)

        table_df = pd.DataFrame(table_data, columns=columns)
        facies_df = assign_lithofacies(table_df)

        # Save the table to an Excel file with UWI and methods in the file name
        uwi = df['UWI'].iloc[0]  # Assuming UWI is the same for all rows in the DataFrame
        file_name = f"./output/{uwi}_electrofacies_mean_log_responses.xlsx"
        facies_df.to_excel(file_name, index=False)
        print(f"Mean log responses of electrofacies saved as '{file_name}'")

        # Assign lithofacies to original df
        for method, curve_name in zip(clustering_methods, curve_names):
            labels = df.loc[not_null_rows, curve_name]
            lithofacies = facies_df.loc[facies_df['Method'] == method].set_index('Cluster')['Lithofacies']

            df.loc[not_null_rows, f'{curve_name}_ASSIGNED'] = labels.map(lithofacies)

        labels = None
        if 'kmeans' in method:
            labels = df.loc[not_null_rows, 'FACIES_KMEANS_ASSIGNED']
        elif 'agglom' in method:
            labels = df.loc[not_null_rows, 'FACIES_AGGLOM_ASSIGNED']
        else:
            labels = df.loc[not_null_rows, f'{curve_name}_ASSIGNED']

        cc = ColorCoding()
        cc.litho_color(lithology_color_coding)
        if masking.get('status'):
            facies_to_drop = masking.get('facies_to_drop')
            label_list = cc.name_to_label(facies_to_drop)
            # Find rows with labels not included in the converted label list
            mask = ~labels.isin(label_list)

        for log, df in zip(logs, dfs):

            uwi = log.well['UWI'].value
            if masking.get('status'):
                for masking_curve in masking.get('curves_to_mask'):
                    data = np.empty(len(log[0]))
                    data[:] = np.nan
                    depth_index = df.loc[(df.UWI == uwi) & mask, 'DEPTH_INDEX']
                    data[depth_index] = df.loc[(df.UWI == uwi) & mask, masking_curve]
                    log.append_curve(
                        f'{masking_curve}_masked',
                        np.copy(data),
                        descr=f'Masked {masking_curve}',
                    )

            for v, vector in enumerate(pc.components_):
                v += 1
                pc_curve = f'PC{v}'

                if pc_curve in log.keys():
                    data = log[pc_curve]
                else:
                    data = np.empty(len(log[0]))
                    data[:] = np.nan
                if masking.get('status'):
                    depth_index = components.loc[(components.UWI == uwi) & mask, 'DEPTH_INDEX']
                    data[depth_index] = np.copy(components.loc[(components.UWI == uwi) & mask, pc_curve])
                else:
                    depth_index = components.loc[components.UWI == uwi, 'DEPTH_INDEX']
                    data[depth_index] = np.copy(components.loc[components.UWI == uwi, pc_curve])

                log.append_curve(
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
                if masking.get('status'):
                    depth_index = df.loc[(df.UWI == uwi) & mask, 'DEPTH_INDEX']
                    data[depth_index] = df.loc[(df.UWI == uwi) & mask, curve_name]
                else:
                    depth_index = df.loc[df.UWI == uwi, 'DEPTH_INDEX']
                    data[depth_index] = df.loc[df.UWI == uwi, curve_name]
                log.append_curve(
                    curve_name,
                    np.copy(data),
                    descr='Electrofacies',
                )

                curve_facies = f'{curve_name}_ASSIGNED'
                if curve_facies in log.keys():
                    data = log[curve_facies]
                else:
                    data = np.empty(len(log[0]))
                    data[:] = np.nan
                if masking.get('status'):
                    depth_index = df.loc[(df.UWI == uwi) & mask, 'DEPTH_INDEX']
                    data[depth_index] = df.loc[(df.UWI == uwi) & mask, curve_facies]
                else:
                    depth_index = df.loc[df.UWI == uwi, 'DEPTH_INDEX']
                    data[depth_index] = df.loc[df.UWI == uwi, curve_facies]
                log.append_curve(
                    curve_facies,
                    np.copy(data),
                    descr=f'Auto-Assigned Lithofacies for {curve_name}',
                )
    if output_template is None:
        return logs
    else:
        return output_template, logs


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


def parsing_membership_track(template_xml_path, curve_names):
    # Load the template file
    tree = ET.parse(template_xml_path)
    root = tree.getroot()

    # Find the track with display_name = "FUZZY MEMBERSHIP" or create a new one if not found
    track_fuzzy_membership = root.find(".//track[@display_name='FUZZY MEMBERSHIP']")
    if track_fuzzy_membership is None:
        track_fuzzy_membership = ET.SubElement(root, "track")
        track_fuzzy_membership.set("display_name", "FUZZY MEMBERSHIP")
        track_fuzzy_membership.set("width", "2")
        track_fuzzy_membership.set("left", "0")
        track_fuzzy_membership.set("right", "1")
        track_fuzzy_membership.set("major_lines", "9")
        track_fuzzy_membership.set("cumulative", "True")
    else:
        # Clear existing curves under the FUZZY MEMBERSHIP track
        existing_curves = track_fuzzy_membership.findall("curve")
        for curve in existing_curves:
            track_fuzzy_membership.remove(curve)

    color_palette = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6",
                     "#6a3d9a", "#ffff99", "#b15928"]

    # Iterate over the membership curves and add them as curves under the FUZZY MEMBERSHIP track
    for i, membership_curve in enumerate(curve_names):
        curve = ET.SubElement(track_fuzzy_membership, "curve")
        curve.set("display_name", f"MEM{i}")
        curve.set("curve_name", membership_curve)
        curve.set("fill_color", color_palette[i % len(color_palette)])

    # Convert the XML tree to a formatted string
    new_xml_str = ET.tostring(track_fuzzy_membership, encoding="utf-8").decode()
    formatted_new_xml_str = minidom.parseString(new_xml_str).toprettyxml(indent="  ")

    # Find the position of the FUZZY MEMBERSHIP track in the original XML tree
    index = list(root).index(track_fuzzy_membership)

    # Replace the XML string of the newly added data with the formatted XML string
    root[index] = ET.fromstring(formatted_new_xml_str)

    # Write the updated XML tree to the original file, overwriting it
    output_template = f'./data/template/electrofacies_{len(curve_names)}members.xml'
    tree.write(output_template, encoding="utf-8", xml_declaration=True)

    return output_template


def assign_lithofacies(df):
    # Calculate median values for CAL_N and SP_N
    cal_median = df['CAL_N'].median()

    # Initialize lithofacies column with 'Unknown' value
    df['Lithofacies'] = 'Unknown'

    for index, row in df.iterrows():
        cluster = row['Cluster']
        cal = row['CAL_N']
        nphi = row['NPHI_N']
        dphi = row['DPHI_N']
        rhob = row['RHOB_N']
        pe = row['PE_N']
        sgr = row['SGR_N']

        if abs(cal - cal_median) > 0.15 * cal_median:
            df.loc[index, 'Lithofacies'] = 99  # 'Anomaly'
        elif nphi < 0.02 and dphi < 0.01 and rhob > 2.7 and pe > 4:
            df.loc[index, 'Lithofacies'] = 9  # 'Anhydrite'
        elif nphi < 0.02 and dphi > 0.3 and rhob < 2.15 and pe > 4:
            df.loc[index, 'Lithofacies'] = 8  # 'Halite'
        elif pe > 3.35 and nphi - dphi > 0.05:
            df.loc[index, 'Lithofacies'] = 7  # 'Dolomite'
        elif pe > 3.5 and nphi - dphi < 0.05:
            df.loc[index, 'Lithofacies'] = 6  # 'Limestone'
        elif rhob < 2.7 and pe < 2.8 and nphi > 0.20 and dphi > 0.15:
            df.loc[index, 'Lithofacies'] = 1  # 'Sandstone'
        elif 3 > pe > 2.7 and sgr > 60:
            df.loc[index, 'Lithofacies'] = 3  # 'Silty Shale'
        elif 3 > pe > 2.7 and sgr < 60:
            df.loc[index, 'Lithofacies'] = 2  # 'Shaly Sandstone'
        elif pe > 3 and sgr > 80:
            df.loc[index, 'Lithofacies'] = 5  # 'Black Shale'
        elif pe > 3 and sgr > 60:
            df.loc[index, 'Lithofacies'] = 4  # 'Shale'
        else:
            df.loc[index, 'Lithofacies'] = 98  # 'Else'

    return df
