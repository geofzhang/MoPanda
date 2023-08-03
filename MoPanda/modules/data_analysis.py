import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.colors as colors
from adjustText import adjust_text
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import importlib
import subprocess
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import os


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
    data_predict['K_calib'] = a1_fit * data_predict[x1_column] ** m1_fit * data_predict[y1_column] ** n1_fit + a2_fit * data_predict[
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


class WellLogPredictor:
    def __init__(self, log=None):
        self.root = tk.Tk()
        self.root.title("Well Log Predictor")

        self.dataframe = log.df()
        self.logs_to_select = []
        self.log_to_predict = None
        self.depth_interval = []

        self.load_dataframe_button = tk.Button(
            self.root, text="Load DataFrame", command=self.load_dataframe_button_click
        )
        self.load_dataframe_button.pack(anchor="center")

        self.select_log_button = tk.Button(self.root, text="Select logs to use", command=self.select_logs)
        self.select_log_button.pack(anchor="center")

        self.select_log_predict_button = tk.Button(self.root, text="Select logs to predict",
                                                   command=self.select_log_to_predict)
        self.select_log_predict_button.pack(anchor="center")

        self.log_interval_button = tk.Button(self.root, text="Select intervals to predict",
                                             command=self.select_depth_interval)
        self.log_interval_button.pack(anchor="center")

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_button_click)
        self.predict_button.pack(anchor="center")

        self.root.mainloop()

    def load_dataframe_button_click(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            try:
                self.dataframe = pd.read_csv(filename, index_col=0)
                messagebox.showinfo("Success", "DataFrame loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def check_dependencies(self, packages):
        not_installed = []
        for package in packages:
            try:
                importlib.import_module(package)
            except ImportError:
                not_installed.append(package)
        return not_installed

    def install_dependencies(self, packages):
        for package in packages:
            subprocess.check_call(["pip", "install", package])

    def check_and_install_dependencies(self):
        required_packages = ["pandas", "scikit-learn", "xgboost", "lightgbm", "catboost"]
        not_installed = self.check_dependencies(required_packages)
        if not_installed:
            messagebox.showinfo("Dependency Check", "Installing required dependencies...")
            self.install_dependencies(not_installed)
            messagebox.showinfo("Dependency Check", "Dependencies installed successfully!")

    def select_logs(self):
        window = tk.Toplevel(self.root)
        window.title("Select Logs")
        window.geometry("300x300")

        selected_logs = []

        def add_log(log):
            selected_logs.append(log)

        def remove_log(log):
            selected_logs.remove(log)

        scrollbar = tk.Scrollbar(window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        log_listbox = tk.Listbox(window, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        log_listbox.pack(fill=tk.BOTH, expand=True)

        for column in self.dataframe.columns:
            log_listbox.insert(tk.END, column)

        scrollbar.config(command=log_listbox.yview)

        search_label = tk.Label(window, text="Search:")
        search_label.pack(anchor="w")

        search_entry = tk.Entry(window)
        search_entry.pack(anchor="w")

        def filter_logs(event=None):
            search_text = search_entry.get().lower()
            log_listbox.delete(0, tk.END)
            for column in self.dataframe.columns:
                if search_text in column.lower():
                    log_listbox.insert(tk.END, column)

        def on_select(event=None):
            selected_logs.clear()
            for index in log_listbox.curselection():
                selected_logs.append(log_listbox.get(index))

        search_entry.bind("<KeyRelease>", filter_logs)
        log_listbox.bind("<<ListboxSelect>>", on_select)

        def apply_selection():
            self.logs_to_select = selected_logs
            window.destroy()

        apply_button = tk.Button(window, text="Apply", command=apply_selection)
        apply_button.pack(anchor="center")

        window.mainloop()

    def select_log_to_predict(self):
        window = tk.Toplevel(self.root)
        window.title("Select Log to Predict")
        window.geometry("300x300")

        selected_log = tk.StringVar()

        def apply_selection():
            self.log_to_predict = selected_log.get()
            window.destroy()

        # Filter out empty or missing column names
        log_options = [log for log in self.dataframe.columns if log.strip()]
        if not log_options:
            messagebox.showerror("Error", "No valid log names available for selection!")
            window.destroy()
            return

        log_dropdown = ttk.Combobox(window, textvariable=selected_log, values=log_options, state="readonly")
        log_dropdown.pack(anchor="center")

        search_label = tk.Label(window, text="Search:")
        search_label.pack(anchor="center")

        search_entry = tk.Entry(window)
        search_entry.pack(anchor="center")

        def filter_logs(event=None):
            search_text = search_entry.get().lower()
            filtered_options = [log for log in log_options if search_text in log.lower()]
            log_dropdown["values"] = filtered_options

        search_entry.bind("<KeyRelease>", filter_logs)

        apply_button = tk.Button(window, text="Apply", command=apply_selection)
        apply_button.pack(anchor="center")

        window.mainloop()

    def select_depth_interval(self):
        window = tk.Toplevel(self.root)
        window.title("Select Depth Interval")
        window.geometry("300x300")

        upper_label = tk.Label(window, text="Top Depth of the interval (ft)")
        upper_label.pack(anchor="center")
        upper_entry = tk.Entry(window)
        upper_entry.pack(anchor="center")

        lower_label = tk.Label(window, text="Bottom Depth of the interval (ft)")
        lower_label.pack(anchor="center")
        lower_entry = tk.Entry(window)
        lower_entry.pack(anchor="center")

        def apply_selection():
            lower_depth = int(lower_entry.get())
            upper_depth = int(upper_entry.get())
            self.depth_interval = [upper_depth, lower_depth]
            window.destroy()

        apply_button = tk.Button(window, text="Apply", command=apply_selection)
        apply_button.pack(anchor="center")

        window.mainloop()

    def train_and_predict(self, model):
        # Filter the dataframe based on the selected depth interval
        df_train = self.dataframe.loc[
            (self.dataframe.index <= self.depth_interval[0]) | (self.dataframe.index >= self.depth_interval[1])
            ].copy()  # Make a copy of the DataFrame

        print(df_train.describe())
        # Drop missing values in the selected logs
        logs_to_dropna = self.logs_to_select + [self.log_to_predict]
        df_train.dropna(subset=logs_to_dropna, inplace=True)

        # Split the data into features (X) and target (y)
        X = df_train[self.logs_to_select]
        y = df_train[self.log_to_predict]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model == "Ridge Regression":
            regressor = Ridge()
        elif model == "Random Forest":
            regressor = RandomForestRegressor()
        elif model == "XGBoost":
            regressor = xgb.XGBRegressor()
        elif model == "LightGBM":
            regressor = lgb.LGBMRegressor()
        elif model == "CatBoost":
            regressor = cb.CatBoostRegressor(silent=True)
        else:
            raise ValueError("Invalid model name")

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return rmse

    def predict_button_click(self):
        self.check_and_install_dependencies()

        if self.dataframe is None or self.dataframe.empty:
            messagebox.showerror("Error", "No DataFrame loaded!")
            return

        if not self.logs_to_select:
            messagebox.showerror("Error", "No logs selected!")
            return

        if not self.log_to_predict:
            messagebox.showerror("Error", "No log selected to predict!")
            return

        if not self.depth_interval:
            messagebox.showerror("Error", "No depth interval selected!")
            return

        models = ["Ridge Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
        results = {}

        # self.select_logs()
        # self.select_log_to_predict()
        # self.select_depth_interval()

        # Display a message box indicating training has started
        messagebox.showinfo("Training Started", "Training started!\n\nDepth Intervals:\n\n" +
                            f"Top Depth: {self.depth_interval[0]} ft\n" +
                            f"Bottom Depth: {self.depth_interval[1]} ft")

        for model in models:
            rmse = self.train_and_predict(model)
            results[model] = rmse

        messagebox.showinfo("Results (RMSE)", "\n".join([f"{model}: {rmse}" for model, rmse in results.items()]))

        selected_model = messagebox.askquestion("Select Model", "Do you want to select a model for prediction?",
                                                icon='question', default='no')

        if selected_model == 'yes':
            model_selection_window = tk.Toplevel(self.root)
            model_selection_window.title("Select Model")
            model_selection_window.geometry("300x200")

            selected_model = tk.StringVar()
            selected_model.set(models[0])  # Set the default selected model

            model_label = tk.Label(model_selection_window, text="Select a model:")
            model_label.pack(anchor="center")

            model_dropdown = ttk.Combobox(model_selection_window, textvariable=selected_model, values=models,
                                          state="readonly")
            model_dropdown.pack(anchor="center")

            def apply_model_selection():
                model_name = selected_model.get()
                regressor = None

                if model_name == "Ridge Regression":
                    regressor = Ridge()
                elif model_name == "Random Forest":
                    regressor = RandomForestRegressor()
                elif model_name == "XGBoost":
                    regressor = xgb.XGBRegressor()
                elif model_name == "LightGBM":
                    regressor = lgb.LGBMRegressor()
                elif model_name == "CatBoost":
                    regressor = cb.CatBoostRegressor(silent=True)
                else:
                    raise ValueError("Invalid model name")

                # Filter the dataframe based on the selected depth interval
                df_train = self.dataframe.loc[
                    (self.dataframe.index <= self.depth_interval[0]) | (self.dataframe.index >= self.depth_interval[1])
                    ].copy()  # Make a copy of the DataFrame

                # Drop missing values in the selected logs
                logs_to_dropna = self.logs_to_select + [self.log_to_predict]
                df_train.dropna(subset=logs_to_dropna, inplace=True)

                # Split the data into features (X) and target (y)
                X_train = df_train[self.logs_to_select]
                y_train = df_train[self.log_to_predict]

                # Train the model on the entire training dataset
                regressor.fit(X_train, y_train)

                # Filter the dataframe based on the selected depth interval for prediction
                df_pred = self.dataframe.loc[
                    (self.dataframe.index >= self.depth_interval[0]) & (self.dataframe.index <= self.depth_interval[1])
                    ].copy()

                # Drop missing values in the selected logs
                logs_to_dropna = self.logs_to_select + [self.log_to_predict]
                df_pred.dropna(subset=logs_to_dropna, inplace=True)

                # Predict the target values using the selected model
                X_pred = df_pred[self.logs_to_select]
                predicted_values = regressor.predict(X_pred)

                print(predicted_values)

                # Add a new column with the predicted values to the dataframe
                new_column_name = f"{self.log_to_predict}_PRE"
                self.dataframe.loc[(self.dataframe.index >= self.depth_interval[0]) & (
                        self.dataframe.index <= self.depth_interval[1]), new_column_name] = predicted_values
                print(self.dataframe[new_column_name])
                # Create a new window to display the results
                result_window = tk.Toplevel(self.root)
                result_window.title("Prediction Results")

                # Create a Treeview widget for displaying the table
                table = ttk.Treeview(result_window, show="headings")
                table.pack(fill="both", expand=True)

                # Define the columns and their headers
                table["columns"] = ("Depth", self.log_to_predict, new_column_name)
                table.heading("Depth", text="Depth")
                table.heading(self.log_to_predict, text=self.log_to_predict)
                table.heading(new_column_name, text=new_column_name)

                # Configure the columns to adjust with the window size
                table.column("Depth", width=100, anchor="center")
                table.column(self.log_to_predict, width=100, anchor="center")
                table.column(new_column_name, width=100, anchor="center")

                # Insert the data into the table
                for index, row in self.dataframe.iterrows():
                    interval = f"{index} ft"
                    value = row[self.log_to_predict]
                    pred_value = row[new_column_name]
                    table.insert("", "end", values=(interval, value, pred_value))

                # Add a scrollbar to the table
                scrollbar = ttk.Scrollbar(result_window, orient="vertical", command=table.yview)
                scrollbar.pack(side="right", fill="y")
                table.configure(yscrollcommand=scrollbar.set)

                messagebox.showinfo("Success", "Predicted values appended to the original dataframe.")
                model_selection_window.destroy()

            apply_model_button = tk.Button(model_selection_window, text="Apply", command=apply_model_selection)
            apply_model_button.pack(anchor="center")

            model_selection_window.mainloop()

        self.logs_to_select = []
        self.log_to_predict = None
        self.depth_interval = []
