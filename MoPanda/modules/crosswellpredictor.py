import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import lasio
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Layer, LSTM, Dense
from keras.callbacks import LambdaCallback, Callback
from keras import backend as K
from keras_self_attention import SeqSelfAttention
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def impute_data(df, selected_logs, logs_to_log_transform):
    """
    Handle missing data by imputing and applying log transformation to selected logs.
    Drop rows from the beginning and the end based on a set of unwanted values
    until a row without those values is encountered.
    """
    unwanted_values = {np.nan, -999.17, -999.25}

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
            df[col] = np.where(df[col] <= 0, 0.000001, df[col])

    # Apply log transformation to selected logs
    for log in logs_to_log_transform:
        if log in df.columns:
            df[log] = np.log(df[log])

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


class SelfAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x, **kwargs):
        mult_data = K.dot(x, self.kernel)
        attention_weights = K.softmax(mult_data)
        output = K.batch_dot(attention_weights, x, axes=[1, 1])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], self.output_dim)

class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])

class CrossWellPredictor(tk.Tk):
    def __init__(self):
        super().__init__()

        # Dictionaries to store the well data. Key = filename, Value = DataFrame of well data.
        self.well_data = None
        self.scaler = None
        self.training_wells = {}
        self.prediction_wells = {}
        self.losses = []

        # Define logs to log-transform
        self.logs_to_log_transform = ['RESDEEP_N', 'K_SDR_N', 'K_GD_N', 'K_TIM_N']

        # Attributes to save the selected logs
        self.selected_x_logs = []
        self.selected_y_log = ""

        # GUI Elements
        self.title("Cross Well Predictor")

        # Importing Training Well Data - Top-left
        self.upload_train_button = tk.Button(self, text="Upload Training Wells",
                                             command=lambda: self.upload_well_data("train"))
        self.upload_train_button.grid(row=1, column=0, pady=5, padx=10)

        self.train_listbox = tk.Listbox(self, height=10, width=60)
        self.train_listbox.grid(row=0, column=0, columnspan=2, pady=5, padx=10)

        self.remove_train_button = tk.Button(self, text="Remove Selected Wells",
                                             command=lambda: self.remove_selected("train"))
        self.remove_train_button.grid(row=1, column=1, pady=5, padx=5)

        # Importing Prediction Well Data - Top-right
        self.upload_predict_button = tk.Button(self, text="Upload Prediction Wells",
                                               command=lambda: self.upload_well_data("predict"))
        self.upload_predict_button.grid(row=1, column=2, pady=5, padx=10)

        self.predict_listbox = tk.Listbox(self, height=10, width=60)
        self.predict_listbox.grid(row=0, column=2, columnspan=2, pady=5, padx=10)

        self.remove_predict_button = tk.Button(self, text="Remove Selected Wells",
                                               command=lambda: self.remove_selected("predict"))
        self.remove_predict_button.grid(row=1, column=3, pady=5, padx=5)

        # Identifying & Selecting Common Logs for X - Bottom-left
        self.identify_logs_button = tk.Button(self, text="Select Training Logs",
                                              command=self.identify_common_logs)
        self.identify_logs_button.grid(row=3, column=0, pady=5, padx=10)

        self.log_selection_listbox = tk.Listbox(self, height=10, width=60, selectmode=tk.MULTIPLE)
        self.log_selection_listbox.grid(row=2, column=0, columnspan=2, pady=5, padx=10)

        self.confirm_x_selection_button = tk.Button(self, text="Confirm", command=self.confirm_x_selection)
        self.confirm_x_selection_button.grid(row=3, column=1, pady=5, padx=10)

        # Identifying & Selecting Common Logs for Y from Training Wells - Bottom-right
        self.identify_y_logs_button = tk.Button(self, text="Select Predicting Log",
                                                command=self.identify_common_y_logs)
        self.identify_y_logs_button.grid(row=3, column=2, pady=5, padx=10)

        self.y_log_selection_listbox = tk.Listbox(self, height=10, width=60, selectmode=tk.SINGLE)
        self.y_log_selection_listbox.grid(row=2, column=2, columnspan=2, pady=5, padx=10)

        self.confirm_y_selection_button = tk.Button(self, text="Confirm", command=self.confirm_y_selection)
        self.confirm_y_selection_button.grid(row=3, column=3, pady=10, padx=10)

        self.train_model_button = tk.Button(self, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=4, column=0, pady=30, padx=10)

        self.predict_button = tk.Button(self, text="Predict for Wells", command=self.predict_for_each_well_display)
        self.predict_button.grid(row=4, column=1, pady=30, padx=10)

        self.progress_text = tk.Text(self, height=20, width=45)
        self.progress_text.grid(row=5, column=0, columnspan=2, pady=5, padx=10)

        self.train_with_hyperparams_button = tk.Button(self, text="Train with Hyperparameters",
                                                       command=self.train_with_hyperparameters)
        self.train_with_hyperparams_button.grid(row=6, column=0, pady=10, padx=10)

        self.save_model_button = tk.Button(self, text="Save Model", command=self.save_model)
        self.save_model_button.grid(row=7, column=0, pady=10, padx=10)

        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=7, column=1, pady=10, padx=10)

    def prepare_data(self):
        # Combine the data of all training wells
        combined_data = pd.concat(list(self.training_wells.values()), ignore_index=True)

        # Handle missing data and apply log transformation to specified logs
        combined_data = impute_data(combined_data, self.selected_x_logs, self.logs_to_log_transform)

        # Calculate the scaler based on all combined training well data
        combined_data, self.scaler = normalize_data(combined_data)

        # Select X and Y data from the combined_data
        X_data = combined_data[self.selected_x_logs].values
        Y_data = combined_data[self.selected_y_log].values

        # Reshape X_data for LSTM
        X_data_reshaped = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])

        # Reshape Y_data
        Y_data_reshaped = Y_data.reshape(-1, 1)

        return X_data_reshaped, Y_data_reshaped

    def upload_well_data(self, mode):
        file_path = filedialog.askopenfilename(filetypes=[("All Supported Types", ".csv .xlsx .las"),
                                                          ("CSV files", "*.csv"),
                                                          ("Excel files", "*.xlsx"),
                                                          ("LAS files", "*.las")])

        if file_path:
            if file_path.endswith('.csv'):
                self.well_data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.well_data = pd.read_excel(file_path)
            elif file_path.endswith('.las'):
                self.well_data = lasio.read(file_path).df

            # Store the data based on the mode (train/predict)
            filename = file_path.split("/")[-1]
            if mode == "train":
                self.training_wells[filename] = self.well_data
                self.train_listbox.insert(tk.END, filename)
            else:
                self.prediction_wells[filename] = self.well_data
                self.predict_listbox.insert(tk.END, filename)

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
                self.predict_listbox.delete(selected)

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

    def build_model(self, input_shape, lstm_units=200, learning_rate=0.01):
        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True, activation='relu'))
        model.add(SeqSelfAttention(attention_activation='sigmoid'))
        model.add(LSTM(lstm_units, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Use linear activation for the output layer

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    def train_model(self):
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, "Training model...\n")

        X_train, Y_train = self.prepare_data()
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = Y_train.shape[1]

        self.model = self.build_model(input_shape)

        # Create an instance of the custom callback
        loss_history_callback = LossHistory()

        # Train the model and provide the custom callback
        self.model.fit(X_train, Y_train, epochs=500, batch_size=64,
                       callbacks=[LambdaCallback(on_epoch_end=self.on_epoch_end),
                                  loss_history_callback])  # Use the custom callback here

        self.progress_text.insert(tk.END, "Training complete!\n")

        # Generate and display the loss vs. epochs plot using the collected loss values
        self.plot_loss(loss_history_callback.losses)

    def plot_loss(self, loss_history):
        loss_fig = plt.figure(figsize=(3, 2))
        plt.plot(loss_history)
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        canvas = FigureCanvasTkAgg(loss_fig, master=self)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().grid(row=5, column=2, columnspan=2, pady=10, padx=10)

    def on_epoch_end(self, epoch, logs):
        msg = f"Epoch {epoch + 1}, Loss: {logs['loss']}\n"
        self.progress_text.insert(tk.END, msg)
        self.progress_text.see(tk.END)
        self.update()

    def predict_for_each_well_display(self):
        predictions = self.predict_for_each_well()
        for well, prediction in predictions.items():
            self.progress_text.insert(tk.END, f"Predictions for {well}:\n")
            self.progress_text.insert(tk.END, f"{prediction}\n")
            self.progress_text.see(tk.END)

    def denormalize_prediction(self, y_pred, scaler):
        """
        Revert normalization of data using provided scaler for prediction values.
        """
        y_pred_denorm = scaler[self.selected_y_log].inverse_transform(y_pred.reshape(-1, 1))

        # Apply inverse log transformation to specified logs
        if self.selected_y_log in self.logs_to_log_transform:
            y_pred_denorm = np.exp(y_pred_denorm)

        return y_pred_denorm

    def predict_for_each_well(self):
        predictions = {}

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
            X_pred_reshaped = X_pred.reshape(X_pred.shape[0], 1, X_pred.shape[1])

            y_pred = self.model.predict(X_pred_reshaped)

            # Denormalize the prediction
            y_pred_denorm = self.denormalize_prediction(y_pred, self.scaler)

            # Convert y_pred_denorm to a DataFrame for easy indexing
            y_pred_df = pd.DataFrame(y_pred_denorm, columns=[self.selected_y_log + '_PRE'])

            # Insert the predicted values back into the original DataFrame using indices
            for i, index in enumerate(original_indices):
                df.loc[index, self.selected_y_log + '_PRE'] = y_pred_df.loc[i, self.selected_y_log + '_PRE']

            # Write to file (if desired)
            df.to_csv(well_name + '.csv', index=False)

            predictions[well_name] = y_pred_denorm

        return predictions

    def train_with_hyperparameters(self):
        X_train, Y_train = self.prepare_data()
        # Example: We'll just tune the number of LSTM units and the learning rate
        best_score = float('inf')  # Assume we are minimizing loss
        best_params = None

        lstm_units_options = [50, 200]
        learning_rates = [0.01, 0.0001]

        for units in lstm_units_options:
            for lr in learning_rates:
                model = self.build_model((X_train.shape[1], X_train.shape[2]), lstm_units=units, learning_rate=lr)
                history = model.fit(X_train, Y_train, epochs=500, batch_size=64, verbose=0)

                # Check final loss for the model
                final_loss = history.history['loss'][-1]

                if final_loss < best_score:
                    best_score = final_loss
                    best_params = (units, lr)

        messagebox.showinfo("Hyperparameter Tuning",
                            f"Best Loss: {best_score}\nBest LSTM Units: {best_params[0]}\nBest Learning Rate: {best_params[1]}")

    def save_model(self):
        if hasattr(self, 'model'):
            file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
            if file_path:
                self.model.save(file_path)
                messagebox.showinfo("Success", "Model saved successfully!")
        else:
            messagebox.showerror("Error", "No trained model found!")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.model = load_model(file_path)
            messagebox.showinfo("Success", "Model loaded successfully!")


# To run the GUI
if __name__ == "__main__":
    app = CrossWellPredictor()
    app.mainloop()
