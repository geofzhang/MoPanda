import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def fit_curve(csv_file, x1_column, y1_column, x2_column, y2_column, z_column):
    # Define the function to fit
    def curve_function(xy, a1, m1, n1, a2, m2, n2):
        x1, y1, x2, y2 = xy
        return a1 * x1 ** m1 * y1 ** n1 + a2 * x2 ** m2 * y2 ** n2

    # Read the data from CSV file
    data = pd.read_csv(csv_file)
    data_predict = data

    # Filter out rows where 'z_column' exists
    data = data.dropna(subset=[z_column])
    # Extract x1, y1, x2, and y2 data from the DataFrame
    x1_data = data[x1_column].values
    y1_data = data[y1_column].values
    x2_data = data[x2_column].values
    y2_data = data[y2_column].values
    z_data = data[z_column].values

    # Set non-negative bounds for the parameters
    bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.inf)

    # Initial guess for the parameters (a, m, n)
    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Perform the curve fit
    params, params_covariance = curve_fit(curve_function, (x1_data, y1_data, x2_data, y2_data), z_data,
                                          p0=initial_guess, bounds=bounds)

    # Extract the fitted parameters
    a1_fit, m1_fit, n1_fit, a2_fit, m2_fit, n2_fit = params

    # Calculate RMSE
    z_pred = curve_function((x1_data, y1_data, x2_data, y2_data), a1_fit, m1_fit, n1_fit, a2_fit, m2_fit, n2_fit)

    rmse = np.sqrt(np.mean((z_pred - z_data) ** 2))
    # Calculate K_calib for all rows
    data_predict['K_calib'] = a1_fit * data_predict[x1_column] ** m1_fit * data_predict[y1_column] ** n1_fit + a2_fit * \
                              data_predict[
                                  x2_column] ** m2_fit * data_predict[y2_column] ** n2_fit

    # Write the updated DataFrame back to CSV
    output_file = os.path.join('../output', os.path.basename(csv_file))
    data_predict.to_csv(output_file, index=False)
    print(
        f"Fitted parameters: a1={a1_fit}, m1={m1_fit}, n1={n1_fit}, a2={a2_fit}, m2={m2_fit}, n2={n2_fit}, RMSE={rmse}")

    return a1_fit, m1_fit, n1_fit, a2_fit, m2_fit, n2_fit, rmse


def fill_null(df):
    for column in df.columns:
        null_counts = df[column].isnull().sum()
        if null_counts > 50:
            # Use polynomial interpolation
            df[column].interpolate(method='polynomial', order=2, inplace=True)
        elif 20 < null_counts <= 50:
            # Use spline interpolation
            null_indices = np.where(df[column].isnull())[0]
            valid_indices = np.where(df[column].notnull())[0]
            if null_indices[0] < valid_indices[0]:
                valid_indices = np.concatenate(([null_indices[0]], valid_indices))
            if null_indices[-1] > valid_indices[-1]:
                valid_indices = np.concatenate((valid_indices, [null_indices[-1]]))
            spline_func = interp1d(valid_indices, df[column].values[valid_indices], kind='cubic')
            df[column].values[null_indices] = spline_func(null_indices)
        else:
            # Use smooth interpolation
            df[column].interpolate(method='linear', inplace=True)

    return df


def plot_pc_crossplot(components, pca_loading, labels, num_top_logs):
    """
    Plot a PC1 and PC2 crossplot with clusters and arrows representing top contributing logs.

    Parameters
    ----------
    components : DataFrame
        DataFrame containing principal components' matrix.
    labels : array-like
        Array of cluster labels.
    num_logs : int or None, optional
        Number of logs to display. If None, all logs will be displayed. Default is None.

    """
    pc1_scores = components.iloc[:, 0]
    pc2_scores = components.iloc[:, 1]

    # Extract the loadings for PC1 and PC2
    pc1_loadings = pca_loading.iloc[0]
    pc2_loadings = pca_loading.iloc[1]

    # Identify the top contributing logs based on loadings
    top_contributors = np.argsort(np.abs(pc1_loadings + pc2_loadings))[::-1]
    top_logs = [pca_loading.columns[i] for i in top_contributors[:num_top_logs]]

    plt.figure(figsize=(12, 12))  # Set the figure size to 20x20

    ax = plt.gca()
    cmap = plt.cm.get_cmap('Set1', len(np.unique(labels)))

    for cluster_label in np.unique(labels):
        mask = (labels == cluster_label)
        ax.scatter(pc1_scores[mask], pc2_scores[mask], c=[cmap(cluster_label - 1)],
                   label=f'Cluster {cluster_label}')

    if top_logs is not None:
        annotations = []
        arrows = []
        for log in top_logs:
            score = np.abs(pc1_loadings[top_logs.index(log)]) + np.abs(pc2_loadings[top_logs.index(log)])
            annotation = ax.annotate(f'{log}\n({score:.2f})',
                                     (pc1_loadings[top_logs.index(log)], pc2_loadings[top_logs.index(log)]),
                                     xytext=(1, -1), textcoords='offset points', fontsize=9, va="center", ha="center", )
            annotations.append(annotation)

            # Add arrow pointing to the annotation
            arrow_x = pc1_loadings[top_logs.index(log)]
            arrow_y = pc2_loadings[top_logs.index(log)]
            arrow = ax.arrow(0, 0, arrow_x, arrow_y, color='red', width=0.01,
                             head_width=0.04, head_length=0.08)
            arrows.append(arrow)

        # Adjust the annotations to minimize overlap
        adjust_text(annotations, objects=arrows)
    min_range = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_range = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')

    ax.legend()
    ax.set_title('Clustering result with relative log importance')
    plt.show(block=False)


def plot_cluster_error(cluster_range, error_scores):
    """
    Plot the number of clusters vs classification error.

    Parameters
    ----------
    cluster_range : range or list
        Range or list of cluster numbers.
    error_scores : list
        List of classification error scores for each cluster number.

    """
    plt.plot(cluster_range, error_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Classification Error')
    plt.title('Number of Clusters vs Classification Error')
    plt.show(block=False)


def plot_pca_variance(variance_ratio):
    """
    Plot the variance ratio of PCA components.

    Parameters
    ----------
    variance_ratio : array-like
        Array of variance ratios for each PCA component.

    """
    n_components = len(variance_ratio)
    cumulative_ratio = np.cumsum(variance_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the variance ratios as a bar plot
    ax1.bar(range(1, n_components + 1), variance_ratio, color='cornflowerblue', alpha=0.5)
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Variance Ratio')

    # Add the cumulative curve
    ax2 = ax1.twinx()
    ax2.plot(range(1, n_components + 1), cumulative_ratio, color='lightcoral')
    ax2.set_ylabel('Cumulative Variance Ratio')

    # Add transparency for columns with cumulative ratio > 0.85
    for i, ratio in enumerate(cumulative_ratio):
        if ratio > 0.85:
            ax1.get_children()[i].set_alpha(0.5)

    # Set x-axis ticks and labels
    ax1.set_xticks(range(1, n_components + 1))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, n_components + 1)])

    plt.title('PCA Variance Ratio and Cumulative Variance')

    plt.show(block=False)


def plot_pca_subplots(components, pca_loading, labels, num_top_logs, variance_ratio):
    """
    Create a plot with two subplots: variance ratio and PC crossplot.

    Parameters
    ----------
    components : DataFrame
        DataFrame containing principal components' matrix.
    pca_loading : DataFrame
        DataFrame containing PCA loadings.
    labels : array-like
        Array of cluster labels.
    num_top_logs : int or None, optional
        Number of logs to display in the PC crossplot. If None, all logs will be displayed. Default is None.
    variance_ratio : array-like
        Array of variance ratios of PCs.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot variance ratio
    ax1.set_box_aspect(1)
    ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=1, color='cornflowerblue')
    ax1.plot(range(1, len(variance_ratio) + 1), np.cumsum(variance_ratio), color='lightcoral')
    threshold = 0.85
    idx = np.argmax(np.cumsum(variance_ratio) > threshold)
    ax1.bar(range(idx + 2, len(variance_ratio) + 1), variance_ratio[idx + 1:], alpha=1, color='silver')
    ax1.axhline(y=threshold, color='gray', linestyle='--')
    ax1.text(len(variance_ratio), threshold, f'{threshold:.0%}', color='gray', ha='right', va='bottom')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Variance Ratio')
    ax1.set_title('Variance Ratio of Principal Components')

    # Plot PC crossplot

    pc1_scores = components.iloc[:, 0]
    pc2_scores = components.iloc[:, 1]

    # Extract the loadings for PC1 and PC2
    pc1_loadings = pca_loading.iloc[0]
    pc2_loadings = pca_loading.iloc[1]

    # Identify the top contributing logs based on loadings
    top_contributors = np.argsort(np.abs(pc1_loadings + pc2_loadings))[::-1]
    top_logs = [pca_loading.columns[i] for i in top_contributors[:num_top_logs]]

    cmap = plt.cm.get_cmap('Set1', len(np.unique(labels)))
    for cluster_label in np.unique(labels):
        mask = (labels == cluster_label)
        ax2.scatter(pc1_scores[mask], pc2_scores[mask], c=[cmap(cluster_label - 1)],
                    label=f'Cluster {cluster_label}')

    if num_top_logs is not None:
        annotations = []
        arrows = []
        for log in top_logs:
            score = np.abs(pc1_loadings[top_logs.index(log)]) + np.abs(pc2_loadings[top_logs.index(log)])
            annotation = ax2.annotate(f'{log}\n({score:.2f})',
                                      (pc1_loadings[top_logs.index(log)], pc2_loadings[top_logs.index(log)]),
                                      xytext=(1, -1), textcoords='offset points', fontsize=12, va="center", ha="center")
            annotations.append(annotation)

            # Add arrow pointing to the annotation
            arrow_x = pc1_loadings[top_logs.index(log)]
            arrow_y = pc2_loadings[top_logs.index(log)]
            arrow = ax2.arrow(0, 0, arrow_x, arrow_y, color='red', width=0.02,
                              head_width=0.08, head_length=0.12)
            arrows.append(arrow)

        # Adjust the annotations to minimize overlap
        adjust_text(annotations, objects=arrows)
    min_range = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    max_range = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.set_xlim(min_range, max_range)
    ax2.set_ylim(min_range, max_range)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Clustering result with relative log importance')
    ax2.legend()

    ax2.set_box_aspect(1)

    plt.tight_layout()
    plt.show(block=False)
