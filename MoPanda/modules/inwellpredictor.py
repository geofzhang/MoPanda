import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import importlib
import subprocess
import catboost as cb
import lightgbm as lgb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class InWellPredictor:
    def __init__(self, log=None):
        self.root = tk.Tk()
        self.root.title("In Well Log Predictor")

        self.filename = None
        self.dataframe = None
        self.logs_to_select = []
        self.log_to_predict = None
        self.depth_interval = []
        self.selected_depth_interval = tk.StringVar()
        self.selected_depth_interval.set("")

        # Entry widget to display the selected file
        self.selected_file_entry = tk.Entry(self.root, width=40)
        self.selected_file_entry.grid(row=0, column=0, columnspan=2, padx=10, pady=5)  # X and Y padding added
        self.selected_file_entry.bind("<FocusOut>", self.update_filename)
        self.selected_file_entry.bind("<Return>", self.update_filename)

        self.load_dataframe_button = tk.Button(
            self.root, text="Load DataFrame", command=self.load_dataframe_button_click
        )
        self.load_dataframe_button.grid(row=0, column=2, padx=10, pady=5)  # X and Y padding added

        self.selected_logs_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, width=40, height=5)
        self.selected_logs_listbox.grid(row=1, column=0, columnspan=2, rowspan=1, padx=10, pady=5)

        self.select_log_button = tk.Button(self.root, text="Select logs to use", command=self.select_logs)
        self.select_log_button.grid(row=1, column=2, columnspan=1, padx=10, pady=5)  # X and Y padding added

        self.selected_log_to_predict_listbox = tk.Listbox(self.root, selectmode=tk.SINGLE, width=40, height=1)
        self.selected_log_to_predict_listbox.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.select_log_predict_button = tk.Button(self.root, text="Select logs to predict",
                                                   command=self.select_log_to_predict)
        self.select_log_predict_button.grid(row=2, column=2, columnspan=1, padx=10, pady=5)  # X and Y padding added
        self.depth_interval_label = tk.Label(self.root, text="Selected Depth Interval:")
        self.depth_interval_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.selected_depth_interval_entry = tk.Entry(self.root, textvariable=self.selected_depth_interval,
                                                      state="readonly")
        self.selected_depth_interval_entry.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        self.log_interval_button = tk.Button(self.root, text="Select intervals to predict",
                                             command=self.select_depth_interval)
        self.log_interval_button.grid(row=4, column=2, columnspan=1, padx=10, pady=5)  # X and Y padding added

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_button_click)
        self.predict_button.grid(row=5, column=2, columnspan=1, padx=10, pady=5)  # X and Y padding added

        self.export_button = tk.Button(self.root, text="Export", command=self.export_dataframe)
        self.export_button.grid(row=6, column=2, columnspan=1, padx=10, pady=5)  # X and Y padding added

        self.root.mainloop()

    def check_dependencies(self, packages):
        not_installed = []
        for package in packages:
            try:
                importlib.import_module(package)
            except ImportError:
                not_installed.append(package)
        return not_installed

    def update_filename(self, event):
        """
        Update the filename attribute when the entry loses focus or Enter is pressed.
        """
        self.filename = self.selected_file_entry.get()

    def install_dependencies(self, packages):
        for package in packages:
            subprocess.check_call(["pip", "install", package])

    def check_and_install_dependencies(self):
        required_packages = ["pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "openpyxl", "pykrige"]
        not_installed = self.check_dependencies(required_packages)
        if not_installed:
            messagebox.showinfo("Dependency Check", "Installing required dependencies...")
            self.install_dependencies(not_installed)
            messagebox.showinfo("Dependency Check", "Dependencies installed successfully!")

    @staticmethod
    def create_cancel_button(window):
        """
        Creates a Cancel button for the given window.
        """
        cancel_button = tk.Button(window, text="Cancel", command=window.destroy)
        cancel_button.pack(anchor="center")
        return cancel_button

    def load_dataframe_button_click(self):
        """
        Loads a dataframe based on user's file selection.
        """
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if self.filename:
            try:
                if self.filename.endswith('.csv'):
                    self.dataframe = pd.read_csv(self.filename, index_col=0)
                elif self.filename.endswith('.xlsx'):
                    if 'Curves' not in pd.ExcelFile(self.filename).sheet_names:
                        messagebox.showerror("Error", "The .xlsx file does not have a 'Curves' sheet.")
                        return
                    self.dataframe = pd.read_excel(self.filename, sheet_name='Curves', index_col=0)
                # Update the selected file entry with the file path
                self.selected_file_entry.delete(0, tk.END)
                self.selected_file_entry.insert(0, self.filename)
                messagebox.showinfo("Success", "DataFrame loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def write_predictions_to_excel(self, predictions, column_name):
        """
        Write the predictions back to the original Excel file.
        """
        try:
            # Read the original Excel file
            xls = pd.ExcelFile(self.filename)
            writer = pd.ExcelWriter(self.filename, engine='xlsxwriter')

            # Loop through all sheets and write them back
            for sheet_name in xls.sheet_names:
                if sheet_name == 'Curves':
                    # Update the dataframe with the new predictions
                    self.dataframe[column_name] = predictions
                    self.dataframe.to_excel(writer, sheet_name=sheet_name, index=True)
                else:
                    sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            writer.save()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write to file: {e}")

    def select_logs(self):
        """
        Opens a window for the user to select logs.
        """
        window = tk.Toplevel(self.root)
        window.title("Select Logs")
        window.geometry("300x300")

        selected_logs = []

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
            print(self.logs_to_select)

            window.destroy()
            # Update the selected logs Listbox only after applying the selection
            self.update_selected_logs_listbox()

        apply_button = tk.Button(window, text="Apply", command=apply_selection)
        apply_button.pack(anchor="center")

        self.create_cancel_button(window)
        window.mainloop()

    def update_selected_logs_listbox(self):
        """
        Update the selected logs Listbox with the selected logs.
        """
        self.selected_logs_listbox.delete(0, tk.END)
        for log in self.logs_to_select:
            self.selected_logs_listbox.insert(tk.END, log)

    def select_log_to_predict(self):
        """
        Opens a window for the user to select a log for prediction.
        """
        window = tk.Toplevel(self.root)
        window.title("Select Log to Predict")
        window.geometry("300x300")

        selected_log = tk.StringVar()

        def apply_selection():
            selected_log_value = selected_log.get()  # Get the selected log value
            self.log_to_predict = selected_log_value  # Set self.log_to_predict
            window.destroy()
            # Update the selected log to predict in the listbox
            self.selected_log_to_predict_listbox.delete(0, tk.END)
            self.selected_log_to_predict_listbox.insert(tk.END, self.log_to_predict)

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

        self.create_cancel_button(window)
        window.mainloop()

    def select_depth_interval(self):
        """
        Opens a window for the user to select depth intervals.
        """
        # Create a Tkinter variable to store the selected depth interval values
        selected_upper_depth = tk.IntVar()
        selected_lower_depth = tk.IntVar()

        window = tk.Toplevel(self.root)
        window.title("Select Depth Interval")
        window.geometry("300x300")

        upper_label = tk.Label(window, text="Top Depth of the interval (ft)")
        upper_label.pack(anchor="center")
        upper_entry = tk.Entry(window, textvariable=selected_upper_depth)
        upper_entry.pack(anchor="center")

        lower_label = tk.Label(window, text="Bottom Depth of the interval (ft)")
        lower_label.pack(anchor="center")
        lower_entry = tk.Entry(window, textvariable=selected_lower_depth)
        lower_entry.pack(anchor="center")

        def apply_selection():
            lower_depth = selected_lower_depth.get()
            upper_depth = selected_upper_depth.get()
            self.depth_interval = [upper_depth, lower_depth]
            # Update the Entry widget with the selected depth interval values
            self.selected_depth_interval.set(f"Top: {upper_depth} ft, Bottom: {lower_depth} ft")
            window.destroy()

        apply_button = tk.Button(window, text="Apply", command=apply_selection)
        apply_button.pack(anchor="center")

        self.create_cancel_button(window)
        window.mainloop()

    def train_and_predict(self, model):
        # Filter the dataframe based on the selected depth interval
        df_train = self.dataframe.loc[
            (self.dataframe.index <= self.depth_interval[0]) | (self.dataframe.index >= self.depth_interval[1])
            ].copy()  # Make a copy of the DataFrame

        # Replace infinite values (inf) with NaN
        df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

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

    def export_dataframe(self):
        """
        Saves the predicted results to the original Excel file.
        """
        try:
            if self.filename.endswith('.csv'):
                self.dataframe.to_csv(self.filename)
            elif self.filename.endswith('.xlsx'):
                self.dataframe.to_excel(self.filename, sheet_name='Curves')
            messagebox.showinfo("Success", "Predicted results saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save DataFrame: {e}")

    def predict_button_click(self):
        """
        Predicts the values based on user selections and displays the result.
        """
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

                # Replace infinite values (inf) with NaN
                df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
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

                # Replace infinite values (inf) with NaN
                df_pred.replace([np.inf, -np.inf], np.nan, inplace=True)

                # Drop missing values in the selected logs
                logs_to_dropna = self.logs_to_select
                df_pred.dropna(subset=logs_to_dropna, inplace=True)

                # Predict the target values using the selected model
                X_pred = df_pred[self.logs_to_select]
                predicted_values = regressor.predict(X_pred)

                # Add a new column with the predicted values to the dataframe
                new_column_name = f"{self.log_to_predict}_PRE"

                self.dataframe[new_column_name] = self.dataframe[self.log_to_predict]

                self.dataframe.loc[(self.dataframe.index >= self.depth_interval[0]) & (
                        self.dataframe.index <= self.depth_interval[1]), new_column_name] = predicted_values

                # Create a new window to display the results
                result_window = tk.Toplevel(self.root)
                result_window.title("Prediction Results")

                # Create a frame to pack both table and scrollbar
                frame = tk.Frame(result_window)
                frame.pack(fill="both", expand=True)

                # Create a Treeview widget for displaying the table
                table = ttk.Treeview(frame, show="headings")

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
                scrollbar = ttk.Scrollbar(frame, orient="vertical", command=table.yview)
                scrollbar.pack(side="right", fill="y")
                table.configure(yscrollcommand=scrollbar.set)

                # Now pack the table after configuring the scrollbar
                table.pack(fill="both", expand=True)

                messagebox.showinfo("Success", "Predicted values appended to the original dataframe.")
                model_selection_window.destroy()

            apply_model_button = tk.Button(model_selection_window, text="Apply", command=apply_model_selection)
            apply_model_button.pack(anchor="center")

            self.create_cancel_button(model_selection_window)
            model_selection_window.mainloop()

        self.logs_to_select = []
        self.log_to_predict = None
        self.depth_interval = []


if __name__ == "__main__":
    predictor = InWellPredictor()
