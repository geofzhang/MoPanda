import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import tensorflow as tf
import tensorflow_probability as tfp
import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import LambdaCallback, Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils import resample
import joblib
from tensorflow_probability.python.layers import DenseVariational
from tqdm import tqdm

# Register the custom layer before loading the model
tf.keras.utils.get_custom_objects()['SeqSelfAttention'] = SeqSelfAttention

# List available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # GPU is available, print GPU information
    for gpu in gpus:
        print("Device name:", gpu.name)
        print("Device type:", gpu.device_type)
else:
    print("No GPU detected.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU device 0

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def impute_data(df, selected_logs, logs_to_log_transform):
    """
    Handle missing data by imputing and applying log transformation to selected logs.
    Drop rows from the beginning and the end based on a set of unwanted values
    until a row without those values is encountered.
    """
    unwanted_values = {-np.inf, np.inf, np.nan, -999.17, -999.25}

    # Drop rows with unwanted values at the beginning
    while df[selected_logs].iloc[0].isin(unwanted_values).any():
        df = df.iloc[1:]

    # Drop rows with unwanted values at the end
    while df[selected_logs].iloc[-1].isin(unwanted_values).any():
        df = df.iloc[:-1]

    # Reset index without dropping it, so the original indices are preserved as a new column
    df = df.reset_index(drop=False).rename(columns={"index": "original_index"})

    # Interpolate for the remaining NaN values
    df = df.interpolate()

    # If any NaN remains (for example, if NaN values are still at the very beginning or end),
    # fill those with 0
    df = df.fillna(0)

    # Apply log transformation to specific columns
    for col in logs_to_log_transform:
        if col in df.columns:
            df[col] = np.where(df[col] <= 0, 0.00001, df[col])

    # Apply log transformation to selected logs
    for log in logs_to_log_transform:
        if log in df.columns:
            df[log] = np.log10(df[log])

    return df


def normalize_data(df, scaler=None):
    """
    Normalize data columns between 0 and 1.
    If scaler is provided (e.g., for unnormalizing), use them.
    Otherwise, fit new scaler.
    Returns normalized dataframe and the scaler.
    """
    if scaler is None:
        scaler = {}

    df_copy = df.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning

    columns_to_normalize = [col for col in df_copy.columns if col != 'original_index']

    for col in columns_to_normalize:
        if col not in scaler:
            scaler[col] = MinMaxScaler(feature_range=(0, 1))
            df_copy.loc[:, col] = scaler[col].fit_transform(df_copy[col].values.reshape(-1, 1))
        else:
            df_copy.loc[:, col] = scaler[col].transform(df_copy[col].values.reshape(-1, 1))

    return df_copy, scaler


def denormalize_data(df, scaler):
    """
    Revert normalization of data using provided scaler.
    """
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    for col in df_copy.columns:
        df_copy.loc[:, col] = scaler[col].inverse_transform(df_copy[col].values.reshape(-1, 1))
    return df_copy


# Segment the data
def segment_data(data, window_size):
    segments = []
    for i in range(len(data) - window_size + 1):
        segment = data[i:i + window_size]
        segments.append(segment)
    return np.array(segments)


def zero_pad(data, target_length, padding_value=0):
    """
    Zero-pads the given data array along axis 0 to reach the target length.
    :param data: np.array, the array to be padded.
    :param target_length: int, the target length.
    :param padding_value: float, the value to pad.
    :return: np.array, zero-padded array.
    """
    pad_size = target_length - data.shape[0]
    if pad_size > 0:
        return np.pad(data, ((0, pad_size),) + tuple((0, 0) for _ in range(data.ndim - 1)), 'constant',
                      constant_values=padding_value)
    return data


# Define a probabilistic layer
def probabilistic_layer(input_shape, units, sample_n):
    return CustomDenseVariational(
        units=units,
        make_posterior_fn=posterior_mean_field,
        make_prior_fn=prior_trainable,
        kl_weight=1 / sample_n,
        input_shape=input_shape
    )


# Define the posterior function
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(
            loc=t[..., :n],
            scale=1e-5 + tf.nn.softplus(c + t[..., n:])
        )),
    ])


# Define the prior function
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(
            loc=t, scale=1)),
    ])


def predict_bnn(model, X_pred, n_samples):
    predictions = []
    for _ in tqdm(range(n_samples), desc="Predicting Uncertainty", unit="Iteration"):
        pred = model.predict(X_pred)
        predictions.append(pred)  # assuming pred.numpy() is the way to extract predictions from the model
    preds_array = np.array(predictions)
    mean_prediction = np.mean(preds_array, axis=0)
    uncertainty = np.std(preds_array, axis=0)
    return mean_prediction, uncertainty


class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])


class LogDetailsDialog(simpledialog.Dialog):
    def body(self, master):
        # Create and place labels, entry fields, and checkbox
        ttk.Label(master, text="Mnemonic:").grid(row=0, sticky=tk.W)
        ttk.Label(master, text="Description:").grid(row=1, sticky=tk.W)
        ttk.Label(master, text="Unit:").grid(row=2, sticky=tk.W)

        self.mnemonic_entry = ttk.Entry(master)
        self.description_entry = ttk.Entry(master)
        self.unit_entry = ttk.Entry(master)

        self.mnemonic_entry.grid(row=0, column=1)
        self.description_entry.grid(row=1, column=1)
        self.unit_entry.grid(row=2, column=1)

        # Checkbox to decide replacement
        self.replace_existing_var = tk.BooleanVar()
        self.replace_existing_log = ttk.Checkbutton(master, text="Replace existing log if it exists",
                                                    variable=self.replace_existing_var)
        self.replace_existing_log.grid(row=3, columnspan=2, sticky=tk.W)

        return self.mnemonic_entry  # Initial focus

    def apply(self):
        # Set the values to be retrieved
        self.mnemonic = self.mnemonic_entry.get()
        self.description = self.description_entry.get()
        self.unit = self.unit_entry.get()
        self.replace_existing = self.replace_existing_var.get()


class CustomDenseVariational(DenseVariational):
    def __init__(self, units, *args, **kwargs):
        super(CustomDenseVariational, self).__init__(units=units, *args, **kwargs)
        self.units = units

    def get_config(self):
        config = super(CustomDenseVariational, self).get_config()
        config.update({
            "units": self.units,
        })
        return config


class CrossWellPredictor(tk.Tk):
    def __init__(self):
        super().__init__()

        # Dictionaries to store the well data. Key = filename, Value = DataFrame of well data.
        self.las_file_paths = None
        self.current_file_type = None
        self.uncertainty_iterations = 100
        self.sample_n = None
        self.well_data = None
        self.scaler = None
        self.training_wells = {}
        self.prediction_wells = {}
        self.las_file_paths = {}
        self.losses = []
        self.window_size = 1
        self.imbalanced_data = False
        self.method = 'LSTM'

        # Define logs to log-transform
        self.logs_to_log_transform = ['RESDEEP_N', 'K_SDR_N', 'K_GD_N', 'K_TIM_N', 'K_SDR', 'K_SDR_PRE', 'RESD_D',
                                      'RESS_D', 'RESSHAL_N']
        self.logs_to_postprocess = ['K_SDR_N', 'K_GD_N', 'K_TIM_N', 'K_SDR', 'K_SDR_PRE']
        self.pe_logs = ['PEF_D', 'PE_N', 'PEF_N']
        # Attributes to save the selected logs
        self.selected_x_logs = []
        self.selected_y_log = ""

        # GUI Elements
        self.title("Cross Well Log Predictor")

        # Importing Training Well Data - Top-left
        self.upload_train_button = tk.Button(self, text="Upload Training Wells",
                                             command=lambda: self.upload_well_data("train"))
        self.upload_train_button.grid(row=1, column=0, pady=5, padx=10)

        self.train_listbox = tk.Listbox(self, height=5, width=60)
        self.train_listbox.grid(row=0, column=0, columnspan=2, pady=5, padx=10)

        self.remove_train_button = tk.Button(self, text="Remove Selected Wells",
                                             command=lambda: self.remove_selected("train"))
        self.remove_train_button.grid(row=1, column=1, pady=5, padx=5)

        # Importing Prediction Well Data - Top-right
        self.upload_predict_button = tk.Button(self, text="Upload Prediction Wells",
                                               command=lambda: self.upload_well_data("predict"))
        self.upload_predict_button.grid(row=1, column=2, pady=5, padx=10)

        self.predict_listbox = tk.Listbox(self, height=5, width=60)
        self.predict_listbox.grid(row=0, column=2, columnspan=2, pady=5, padx=10)

        self.remove_predict_button = tk.Button(self, text="Remove Selected Wells",
                                               command=lambda: self.remove_selected("predict"))
        self.remove_predict_button.grid(row=1, column=3, pady=5, padx=5)

        # Identifying & Selecting Common Logs for X - Bottom-left
        self.identify_logs_button = tk.Button(self, text="Select Training Logs",
                                              command=self.identify_common_logs)
        self.identify_logs_button.grid(row=3, column=0, pady=5, padx=10)

        self.log_selection_listbox = tk.Listbox(self, height=7, width=60, selectmode=tk.MULTIPLE)
        self.log_selection_listbox.grid(row=2, column=0, columnspan=2, pady=5, padx=10)

        self.confirm_x_selection_button = tk.Button(self, text="Confirm", command=self.confirm_x_selection)
        self.confirm_x_selection_button.grid(row=3, column=1, pady=5, padx=10)

        # Identifying & Selecting Common Logs for Y from Training Wells - Bottom-right
        self.identify_y_logs_button = tk.Button(self, text="Select Predicting Log",
                                                command=self.identify_common_y_logs)
        self.identify_y_logs_button.grid(row=3, column=2, pady=5, padx=10)

        self.y_log_selection_listbox = tk.Listbox(self, height=7, width=60, selectmode=tk.SINGLE)
        self.y_log_selection_listbox.grid(row=2, column=2, columnspan=2, pady=5, padx=10)

        self.confirm_y_selection_button = tk.Button(self, text="Confirm", command=self.confirm_y_selection)
        self.confirm_y_selection_button.grid(row=3, column=3, pady=10, padx=10)

        self.correlation_analysis_button = tk.Button(self, text="Correlation Analysis",
                                                     command=self.calculate_correlations)
        self.correlation_analysis_button.grid(row=4, column=0, pady=5, padx=10)

        self.calculate_uncertainty = tk.BooleanVar(value=False)
        self.checkbox = tk.Checkbutton(self, text="Calculate Uncertainty", variable=self.calculate_uncertainty)
        self.checkbox.grid(row=4, column=1, pady=5, padx=10)

        self.train_model_button = tk.Button(self, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=4, column=2, pady=10, padx=10)

        self.predict_button = tk.Button(self, text="Predict for Wells", command=self.predict_for_each_well_display)
        self.predict_button.grid(row=4, column=3, pady=10, padx=10)

        self.progress_text = tk.Text(self, height=20, width=45)
        self.progress_text.grid(row=5, column=0, columnspan=2, pady=5, padx=10)

        self.train_with_hyperparams_button = tk.Button(self, text="Train with Hyperparameters",
                                                       command=self.train_with_hyperparameters)
        self.train_with_hyperparams_button.grid(row=6, column=0, pady=10, padx=10)

        self.save_model_button = tk.Button(self, text="Save Model", command=self.save_model)
        self.save_model_button.grid(row=7, column=0, pady=10, padx=10)

        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=7, column=1, pady=10, padx=10)

        # Entry widgets for hyperparameters
        self.lstm_units_entry = tk.Entry(self)
        self.lstm_units_entry.grid(row=6, column=2, pady=10, padx=10)
        self.lstm_units_entry.insert(0, "200")  # Default value

        self.learning_rate_entry = tk.Entry(self)
        self.learning_rate_entry.grid(row=6, column=3, pady=10, padx=10)
        self.learning_rate_entry.insert(0, "0.01")  # Default value

        self.dropout_rate_entry = tk.Entry(self)
        self.dropout_rate_entry.grid(row=7, column=2, pady=10, padx=10)
        self.dropout_rate_entry.insert(0, "0.2")  # Default value

        self.apply_hyperparams_button = tk.Button(self, text="Apply Hyperparameters",
                                                  command=self.apply_hyperparameters)
        self.apply_hyperparams_button.grid(row=7, column=3, columnspan=2, pady=10, padx=10)

    def retrieve_hyperparameters(self):
        """
        Retrieve LSTM units, learning rate, and dropout rate specified by the user.
        """
        lstm_units = int(self.lstm_units_entry.get())
        learning_rate = float(self.learning_rate_entry.get())
        dropout_rate = float(self.dropout_rate_entry.get())  # Add this line to retrieve dropout rate
        return lstm_units, learning_rate, dropout_rate

    def process_y_pred_denorm(self, y_pred_denorm):
        # Assign 0 to inf values
        y_pred_denorm = np.where(np.isinf(y_pred_denorm), np.nan, y_pred_denorm)

        if self.selected_y_log in self.logs_to_postprocess:
            # Handle values below 0.00001 and above 1000000
            y_pred_denorm = np.where(y_pred_denorm < 0.00001, 0.00001, y_pred_denorm)
            y_pred_denorm = np.where(y_pred_denorm > 100000, 0.00001, y_pred_denorm)

            # Remove values above 10k
            y_pred_denorm = np.where(y_pred_denorm > 5000, np.nan, y_pred_denorm)
            print("You are predicting Resistivity or Permeability log.")

        if self.selected_y_log in self.pe_logs:
            y_pred_denorm = np.where(y_pred_denorm < 1, np.nan, y_pred_denorm)
            y_pred_denorm = np.where(y_pred_denorm > 8, np.nan, y_pred_denorm)
            print("You are predicting PE log.")

        # Interpolate removed values
        y_pred_denorm = np.interp(np.arange(len(y_pred_denorm)), np.where(~np.isnan(y_pred_denorm))[0],
                                  y_pred_denorm[~np.isnan(y_pred_denorm)])

        # # Apply moving average with a window of 5
        # y_pred_denorm = np.convolve(y_pred_denorm, np.ones(4) / 4, mode='same')

        return y_pred_denorm

    def prepare_data(self):
        # Combine the data of all training wells
        combined_data = pd.concat(list(self.training_wells.values()), ignore_index=True)
        if self.imbalanced_data:
            # Identify imbalanced data and perform downsampling for Y < 0.01
            minority_class = combined_data[combined_data[self.selected_y_log] > 0.1]
            majority_class = combined_data[combined_data[self.selected_y_log] <= 0.1]

            # Downsample the majority class to match the size of the minority class
            majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class),
                                            random_state=42)

            # Combine the minority and downsampled majority classes
            combined_data = pd.concat([majority_downsampled, minority_class])

        # Handle missing data and apply log transformation to specified logs
        combined_data = impute_data(combined_data, self.selected_x_logs, self.logs_to_log_transform)

        # Calculate the scaler based on the combined data
        combined_data, self.scaler = normalize_data(combined_data)

        # Split the data into a training set and a test set
        train_data, test_data = train_test_split(combined_data, test_size=0.3, random_state=42)

        # Select X and Y data from the datasets
        X_train = train_data[self.selected_x_logs].values
        X_test = test_data[self.selected_x_logs].values

        Y_train = train_data[self.selected_y_log].values
        Y_test = test_data[self.selected_y_log].values

        X_train = segment_data(X_train, self.window_size)
        X_test = segment_data(X_test, self.window_size)

        Y_train = Y_train[self.window_size - 1:]  # Match the Y labels to the last sample in each segment
        Y_test = Y_test[self.window_size - 1:]

        return X_train, Y_train, X_test, Y_test

    def upload_well_data(self, mode):
        file_paths = filedialog.askopenfilenames(filetypes=[("All Supported Types", ".csv .xlsx .las"),
                                                            ("CSV files", "*.csv"),
                                                            ("Excel files", "*.xlsx"),
                                                            ("LAS files", "*.las")])

        if not file_paths:
            return

        # Check if all selected files have the same extension
        file_extensions = {os.path.splitext(path)[1] for path in file_paths}
        if len(file_extensions) > 1:
            messagebox.showerror("File Type Error", "Please select files of the same type.")
            return

        # Store the file type (assumes only one type due to previous check)
        self.current_file_type = file_extensions.pop()

        for file_path in file_paths:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            print(filename)
            if file_path.endswith('.csv'):
                self.well_data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.well_data = pd.read_excel(file_path, sheet_name=1)
            elif file_path.endswith('.las'):
                self.well_data = lasio.read(file_path).df().reset_index()
                self.las_file_paths[filename] = file_path

            # Store the data based on the mode (train/predict)

            if mode == "train":
                self.training_wells[filename] = self.well_data
                self.train_listbox.insert(tk.END, filename)
            else:
                self.prediction_wells[filename] = self.well_data
                self.predict_listbox.insert(tk.END, filename)

    def remove_selected(self, mode):
        if mode == "train":
            selected = self.train_listbox.curselection()
            if selected:
                filename = self.train_listbox.get(selected)
                del self.training_wells[filename]
                self.train_listbox.delete(selected)
        elif mode == "predict":
            selected = self.predict_listbox.curselection()
            if selected:
                filename = self.predict_listbox.get(selected)
                del self.prediction_wells[filename]

                # Remove from las_file_paths if it exists there
                if filename in self.las_file_paths:
                    del self.las_file_paths[filename]

                self.predict_listbox.delete(selected)

    def prompt_log_details(self):
        """
        Prompt the user for mnemonic, description, unit, and replacement option of the log in a single dialog.
        """
        dialog = LogDetailsDialog(self.master, title="Enter Log Details")

        # Use the details from the dialog
        return dialog.mnemonic, dialog.description, dialog.unit, dialog.replace_existing

    def identify_common_logs(self):
        # Get all column sets from training and prediction wells
        all_columns = [set(df.columns) for df in self.training_wells.values()] + [set(df.columns) for df in
                                                                                  self.prediction_wells.values()]

        # Find the common columns across all wells
        common_logs = set.intersection(*all_columns)

        if not common_logs:
            messagebox.showerror("Error", "No common logs found!")
            return

        # Clear the listbox and add the common logs
        self.log_selection_listbox.delete(0, tk.END)
        for log in common_logs:
            self.log_selection_listbox.insert(tk.END, log)

    def identify_common_y_logs(self):
        # Get all column sets from just training wells
        all_columns = [set(df.columns) for df in self.training_wells.values()]

        # Find the common columns across all training wells
        common_logs = set.intersection(*all_columns)

        if not common_logs:
            messagebox.showerror("Error", "No common logs found in training wells!")
            return

        # Clear the listbox and add the common logs for Y
        self.y_log_selection_listbox.delete(0, tk.END)
        for log in common_logs:
            self.y_log_selection_listbox.insert(tk.END, log)

    def confirm_x_selection(self):
        selected_logs = [self.log_selection_listbox.get(i) for i in self.log_selection_listbox.curselection()]
        if selected_logs:
            self.selected_x_logs = selected_logs
            messagebox.showinfo("Info", f"Selected logs for X: {', '.join(selected_logs)}")
        else:
            messagebox.showwarning("Warning", "No logs selected for X dataset.")

    def confirm_y_selection(self):
        selected_log = self.y_log_selection_listbox.get(self.y_log_selection_listbox.curselection())
        if selected_log:
            self.selected_y_log = selected_log
            messagebox.showinfo("Info", f"Selected log for Y: {selected_log}")
        else:
            messagebox.showwarning("Warning", "No log selected for Y dataset.")

    def calculate_correlations(self):
        """
        Calculate Pearson correlation coefficients between common X logs and the selected target log.
        """
        if not self.selected_y_log:
            messagebox.showwarning("Warning", "No target log selected for correlation analysis.")
            return

        correlations = {}  # Dictionary to store correlation coefficients

        for log in self.selected_x_logs:
            if log != self.selected_y_log:
                # Initialize lists to store data for correlation analysis
                common_logs_data = []
                y_data = []

                # Extract the log data for correlation analysis
                for well_name, df in self.training_wells.items():
                    if log in df.columns and self.selected_y_log in df.columns:
                        # Extract the data for the common log and target log
                        x_log_data = df[log].values
                        y_log_data = df[self.selected_y_log].values

                        # Define unwanted values
                        unwanted_values = {-999.25, -999.17, 999.25, 999.17}

                        # Exclude rows with NaN, inf or unwanted values
                        valid_indices = (np.isfinite(x_log_data) & np.isfinite(y_log_data) &
                                         ~np.isin(x_log_data, unwanted_values) &
                                         ~np.isin(y_log_data, unwanted_values))
                        if np.any(valid_indices):
                            common_logs_data.append(x_log_data[valid_indices])
                            y_data.append(y_log_data[valid_indices])

                # Calculate the Pearson correlation coefficient if there is valid data
                if common_logs_data:
                    common_logs_data = np.concatenate(common_logs_data)
                    y_data = np.concatenate(y_data)
                    print(f"Size of dataset for {log}: {len(common_logs_data)}")
                    print(f"Size of y_data for {log}: {len(common_logs_data)}")

                    correlation_coefficient = pearsonr(common_logs_data, y_data)[0]
                    correlations[log] = correlation_coefficient

        # Sort the logs by correlation coefficient in descending order
        sorted_correlations = {k: v for k, v in
                               sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)}

        # Display or store the sorted correlations
        print("Correlation coefficients with the target log:")
        for log, correlation in sorted_correlations.items():
            print(f"{log}: {correlation:.2f}")

    def build_model(self, input_shape, lstm_units=200, learning_rate=0.01, dropout_rate=0.2):
        model = Sequential()
        print('Algorithm using in the Neural Network:', self.method)

        if self.method == "CNN-LSTM":
            # Add 1D Convolutional layers
            model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(pool_size=1))
            # model.add(Conv1D(128, kernel_size=3, activation='relu'))
            # model.add(MaxPooling1D(pool_size=2))

            # First LSTM layer with Bidirectional wrapper and tanh activation
            model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, activation='tanh')))
            model.add(Dropout(dropout_rate))  # Add dropout layer

            model.add(SeqSelfAttention())

            # Second LSTM layer with Bidirectional wrapper and tanh activation
            model.add(Bidirectional(LSTM(lstm_units, activation='tanh')))

        elif self.method == "LSTM":
            model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True, activation='tanh'))
            model.add(Dropout(dropout_rate))  # Add dropout layer
            model.add(SeqSelfAttention())
            model.add(LSTM(lstm_units, activation='tanh'))

        if self.calculate_uncertainty.get():
            model.add(probabilistic_layer(input_shape, 1, self.sample_n))
        else:
            model.add(Dense(1, activation='linear'))  # Use linear activation for the output layer

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    def train_model(self):
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, "Training model...\n")

        X_train, Y_train, X_test, Y_test = self.prepare_data()
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.sample_n = X_train.shape[0]

        # Create output folder
        current_time = datetime.now().strftime('%m%d%H%M')
        output_folder = os.path.join('./output/CrossWellPrediction/', f'Training Data_{self.selected_y_log}_{current_time}')
        os.makedirs(output_folder, exist_ok=True)

        # Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        # Define model checkpoint
        model_filepath = os.path.join(output_folder, 'best_model.keras')
        model_checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', save_best_only=True, mode='min')

        self.model = self.build_model(input_shape=input_shape)

        # Create an instance of the custom callback
        loss_history_callback = LossHistory()

        # Train the model with early stopping and model checkpoint
        history = self.model.fit(
            X_train, Y_train,
            epochs=500,
            batch_size=64,
            validation_data=(X_test, Y_test),  # Validation set
            callbacks=[LambdaCallback(on_epoch_end=self.on_epoch_end),
                       loss_history_callback,
                       early_stopping,
                       model_checkpoint]
        )

        self.progress_text.insert(tk.END, "Training complete!\n")

        # Generate and display the loss vs. epochs plot using the collected loss values
        self.plot_loss(history.history['loss'], history.history['val_loss'])

        # Save the loss values to a CSV file
        loss_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })

        filename = f'loss_values_{current_time}.csv'
        file_path = os.path.join(output_folder, filename)

        loss_df.to_csv(file_path, index=False)

    def plot_loss(self, training_loss, validation_loss):
        loss_fig = plt.figure(figsize=(6, 4))  # Adjust the figsize as needed
        plt.plot(training_loss, label='Training Loss')
        plt.plot(validation_loss, label='Validation Loss')
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Use sticky to make the figure expand to fill the available space
        canvas = FigureCanvasTkAgg(loss_fig, master=self)
        canvas.get_tk_widget().grid(row=5, column=2, columnspan=2, rowspan=1, pady=10, padx=10, sticky='nsew')

        # Configure row and column weights for proper resizing
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)

        canvas.draw()

    def on_epoch_end(self, epoch, logs):
        msg = f"Epoch {epoch + 1}, Loss: {logs['loss']}\n"
        self.progress_text.insert(tk.END, msg)
        self.progress_text.see(tk.END)
        self.update()

    def predict_for_each_well_display(self):
        if self.calculate_uncertainty.get():
            print("Calculating uncertainty after prediction.")
        predictions, lower_bounds, upper_bounds = self.predict_for_each_well()
        for well_name in predictions.keys():
            prediction = predictions[well_name]
            self.progress_text.insert(tk.END, f"Predictions for {well_name}:\n Completed!")
            self.progress_text.insert(tk.END, f"{prediction}\n")
            self.progress_text.see(tk.END)

    def denormalize_prediction(self, y_pred, scaler):
        """
        Revert normalization of data using provided scaler for prediction values.
        """
        y_pred_denorm = scaler[self.selected_y_log].inverse_transform(y_pred.reshape(-1, 1))

        # Apply inverse log transformation to specified logs
        if self.selected_y_log in self.logs_to_log_transform:
            y_pred_denorm = np.clip(y_pred_denorm, -5, 4)  # Adjust the range as needed
            y_pred_denorm = np.power(10, y_pred_denorm)

        return y_pred_denorm

    def apply_imputation_conditions(self, df, original_indices):
        for i, index in enumerate(original_indices):
            if 'NPHI_N' in df.columns and 'DPHI_N' in df.columns:
                # Check if POR_N exists and if it's equal to 0
                if ('POR_N' in df.columns and df.loc[index, 'POR_N'] == 0) or \
                        ('POR_N' not in df.columns and 'POR_D' in df.columns and df.loc[index, 'POR_D'] == 0):
                    df.loc[index, self.selected_y_log + '_PRE'] = 0.00001
                elif (df.loc[index, 'NPHI_N'] < 0 or df.loc[index, 'DPHI_N'] < 0 or
                      (df.loc[index, 'DPHI_N'] - df.loc[index, 'NPHI_N']) > 0.14):
                    df.loc[index, self.selected_y_log + '_PRE'] = 0.00001
            elif 'NPHI_D' in df.columns and 'DPHI_D' in df.columns:
                # Check if POR_D exists and if it's equal to 0
                if ('POR_D' in df.columns and df.loc[index, 'POR_D'] == 0) or \
                        ('POR_D' not in df.columns and 'POR_N' in df.columns and df.loc[index, 'POR_N'] == 0):
                    df.loc[index, self.selected_y_log + '_PRE'] = 0.00001
                elif (df.loc[index, 'NPHI_D'] < 0 or df.loc[index, 'DPHI_D'] < 0 or
                      (df.loc[index, 'DPHI_D'] - df.loc[index, 'NPHI_D']) > 0.14):
                    df.loc[index, self.selected_y_log + '_PRE'] = 0.00001
        pass

    def predict_for_each_well(self):
        predictions = {}
        well_names = []
        lower_bounds = {}
        upper_bounds = {}

        # Create output folder
        current_time = datetime.now().strftime('%m%d%H%M')
        output_directory = os.path.join('./output/CrossWellPrediction/', f'Predictions_{self.selected_y_log}_{current_time}')
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        las_wells = set(self.las_file_paths.keys())
        predicting_wells = set(self.prediction_wells.keys())
        print('las_wells', las_wells)
        print('predicting_wells', predicting_wells)
        # Check if any of the wells' original file was a .las and prompt for details
        if any(well_name in self.las_file_paths for well_name in self.prediction_wells.keys()):
            mnemonic, description, unit, replace_existing = self.prompt_log_details()

        for well_name, df in self.prediction_wells.items():
            # Handle missing data
            df_imputed = impute_data(df, self.selected_x_logs, self.logs_to_log_transform)
            original_indices = df_imputed["original_index"].tolist()

            # Drop the "original_index" column from df_imputed
            df_imputed = df_imputed.drop(columns=["original_index"])

            # Normalize the data using the new scalers
            df_imputed[self.selected_x_logs], self.scaler = normalize_data(df_imputed[self.selected_x_logs],
                                                                           self.scaler)
            X_pred = df_imputed[self.selected_x_logs].values

            # Zero padding and reshaping the prediction matrix
            X_pred_reshaped = segment_data(X_pred, self.window_size)
            X_pred_reshaped = zero_pad(X_pred_reshaped, len(X_pred))

            if self.calculate_uncertainty.get():
                # Main prediction with uncertainty
                y_pred_mean, y_pred_stddev = predict_bnn(self.model, X_pred_reshaped, self.uncertainty_iterations)
                y_pred_mean = self.denormalize_prediction(y_pred_mean, self.scaler)
                y_pred_mean = self.process_y_pred_denorm(y_pred_mean).reshape(-1, 1)
                y_pred_stddev = self.denormalize_prediction(y_pred_stddev, self.scaler)
                print(y_pred_stddev)

                # Uncertainty estimation
                lower_bound = y_pred_mean - 2 * y_pred_stddev
                upper_bound = y_pred_mean + 2 * y_pred_stddev

                # Convert lower_bound_denorm and upper_bound_denorm to DataFrames
                lower_bound_df = pd.DataFrame(lower_bound, columns=[self.selected_y_log + '_LOWER_BOUND'])
                upper_bound_df = pd.DataFrame(upper_bound, columns=[self.selected_y_log + '_UPPER_BOUND'])

                for i, index in enumerate(original_indices):
                    df.loc[index, self.selected_y_log + '_LOWER_BOUND'] = lower_bound_df.loc[
                        i, self.selected_y_log + '_LOWER_BOUND']
                    df.loc[index, self.selected_y_log + '_UPPER_BOUND'] = upper_bound_df.loc[
                        i, self.selected_y_log + '_UPPER_BOUND']
            else:
                # Main prediction without uncertainty
                y_pred = self.model.predict(X_pred_reshaped)
                y_pred_denorm = self.denormalize_prediction(y_pred, self.scaler)
                y_pred_mean = self.process_y_pred_denorm(y_pred_denorm)

            # Convert y_pred_denorm to DataFrame
            y_pred_df = pd.DataFrame(y_pred_mean, columns=[self.selected_y_log + '_PRE'])

            # Insert the predicted values back into the original DataFrame using indices
            for i, index in enumerate(original_indices):
                df.loc[index, self.selected_y_log + '_PRE'] = y_pred_df.loc[i, self.selected_y_log + '_PRE']

            # Apply imputation based on conditions
            if self.selected_y_log in self.logs_to_postprocess:
                self.apply_imputation_conditions(df, original_indices)

            if well_name in self.las_file_paths:

                # Read original LAS file from the stored path
                las = lasio.read(self.las_file_paths[well_name])
                if replace_existing and self.selected_y_log in las.keys():
                    # Replace the original log
                    las[self.selected_y_log] = df[self.selected_y_log + '_PRE'].values
                    if unit:
                        las.curves[self.selected_y_log].unit = unit
                    if description:
                        las.curves[self.selected_y_log].descr = description
                else:
                    # Add the predicted log
                    las.append_curve(mnemonic, df[self.selected_y_log + '_PRE'].values, unit=unit, descr=description)

                # Save the LAS file with the predictions
                output_file_path = os.path.join(output_directory, well_name + '_Predicted.las')
                las.write(output_file_path)

            else:
                # Define the file path for saving the prediction
                output_file_path = os.path.join(output_directory, well_name + '_Predicted.csv')

                # Write the DataFrame to the specified file path
                df.to_csv(output_file_path, index=False)

            print("Prediction output to:", output_file_path)

            # Append the results to dictionaries
            predictions[well_name] = y_pred_mean
            if self.calculate_uncertainty.get():
                lower_bounds[well_name] = lower_bound
                upper_bounds[well_name] = upper_bound
                self.visualize_uncertainty(predictions[well_name], lower_bounds[well_name], upper_bounds[well_name],
                                           well_name)

        return predictions, lower_bounds, upper_bounds if self.calculate_uncertainty.get() else predictions

    def visualize_uncertainty(self, y_pred, lower_bound, upper_bound, well_name):
        lower_bound = np.squeeze(lower_bound)
        upper_bound = np.squeeze(upper_bound)
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred, label='Predicted', color='blue')
        plt.fill_between(np.arange(len(y_pred)),
                         lower_bound,
                         upper_bound,
                         color='gray', alpha=0.5, label='Uncertainty')
        plt.title(f'Uncertainty Analysis for {well_name}')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Value')
        plt.yscale('log')
        plt.legend()
        plt.show()

    def train_with_hyperparameters(self):
        X_train, Y_train, X_test, Y_test = self.prepare_data()

        # Define hyperparameter search space
        lstm_units_options = [50, 100, 200]
        learning_rates = [0.01, 0.001, 0.0001]
        dropout_rates = [0.0, 0.2, 0.4]

        best_score = float('inf')  # Assume we are minimizing loss
        best_params = None

        for units in lstm_units_options:
            for lr in learning_rates:
                for dropout_rate in dropout_rates:
                    model = self.build_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        lstm_units=units,
                        learning_rate=lr,
                        dropout_rate=dropout_rate
                    )
                    history = model.fit(
                        X_train, Y_train,
                        epochs=500,
                        batch_size=64,
                        validation_data=(X_test, Y_test),  # Validation set
                        verbose=0
                    )

                    # Check final validation loss for the model
                    final_loss = history.history['val_loss'][-1]

                    if final_loss < best_score:
                        best_score = final_loss
                        best_params = (units, lr, dropout_rate)

        # Display the best hyperparameters
        messagebox.showinfo("Hyperparameter Tuning",
                            f"Best Validation Loss: {best_score:.4f}\n"
                            f"Best LSTM Units: {best_params[0]}\n"
                            f"Best Learning Rate: {best_params[1]}\n"
                            f"Best Dropout Rate: {best_params[2]}")

    def apply_hyperparameters(self):
        """
        Apply the specified hyperparameters to build the model and predict using the target log.
        """
        lstm_units, learning_rate, dropout_rate = self.retrieve_hyperparameters()

        # Prepare data
        X_train, Y_train, X_test, Y_test = self.prepare_data()
        input_shape = (X_train.shape[1], X_train.shape[2])

        # Build the model using the specified hyperparameters
        model = self.build_model(input_shape, lstm_units=lstm_units, learning_rate=learning_rate,
                                 dropout_rate=dropout_rate)

        # Train the model
        model.fit(X_train, Y_train, epochs=500, batch_size=64, verbose=0)

        # Perform prediction using the trained model
        predictions, lower_bounds, upper_bounds = self.predict_for_each_well()

        for well_name in predictions.keys():
            prediction = predictions[well_name]
            self.progress_text.insert(tk.END, f"Predictions for {well_name}:\n Completed!")
            self.progress_text.insert(tk.END, f"{prediction}\n")
            self.progress_text.see(tk.END)

    def save_model(self):
        if hasattr(self, 'model'):
            file_path = filedialog.asksaveasfilename(defaultextension=".keras", filetypes=[("Keras files", "*.keras")])
            if file_path:
                # Ensure the file path has the .keras extension
                if not file_path.endswith(".keras"):
                    file_path += ".keras"

                # Save the model using the native Keras format
                self.model.save(file_path)

                # Save the scaler as a separate file with "_scaler.pkl" extension
                scaler_path = file_path.replace(".keras", "_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

                messagebox.showinfo("Success", "Model and scaler saved successfully!")
        else:
            messagebox.showerror("Error", "No trained model found!")

    def load_model(self):
        # Load the model
        file_path = filedialog.askopenfilename(filetypes=[("keras files", "*.keras")])
        if file_path:
            self.model = load_model(file_path)
            messagebox.showinfo("Success", "Model loaded successfully!")

            # Load the scaler using joblib
            scaler_path = file_path.replace(".keras",
                                            "_scaler.pkl")
            try:
                self.scaler = joblib.load(scaler_path)
            except FileNotFoundError:
                messagebox.showerror("Error", "Scaler file not found. Please make sure the scaler file is available.")

            if hasattr(self, 'model') and hasattr(self, 'scaler'):
                messagebox.showinfo("Success", "Model and scaler loaded successfully!")
            else:
                messagebox.showerror("Error", "Model or scaler loading failed.")


# To run the GUI
if __name__ == "__main__":
    app = CrossWellPredictor()
    app.mainloop()
