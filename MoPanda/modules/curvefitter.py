import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import least_squares


class CMRPermeabilityApp(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky=(tk.N, tk.W, tk.E, tk.S))
        self.current_file_path = None  # Store the current file path
        self.setup_gui()

    def setup_gui(self):
        control_frame = tk.Frame(self)
        control_frame.pack(pady=20)

        self.file_path_entry = tk.Entry(self, width=50)
        self.file_path_entry.pack(pady=10)

        self.equations_frame = tk.Frame(self)
        self.equations_frame.pack(pady=20)

        open_btn = tk.Button(control_frame, text="Open Excel File",
                             command=lambda: self.browse_excel_file())
        open_btn.pack(side=tk.LEFT, padx=10)

        format_label = tk.Label(control_frame, text="Format", bg="lightgray")
        format_label.bind("<Enter>", self.show_format_table)
        format_label.pack(side=tk.LEFT, padx=10)

        self.fitting_method = tk.StringVar(value='Absolute Residual')
        self.fitting_method.trace_add('write', self.update_weight_visibility)

        # Adding Combobox for selecting the fitting method
        ttk.Label(self, text="Fitting Method:").pack(pady=(20, 0))
        fitting_method_dropdown = ttk.Combobox(self, textvariable=self.fitting_method,
                                               values=['Absolute Residual', 'Log Transformation', 'Weighted', 'Error Normalization'])
        fitting_method_dropdown.pack(pady=(0, 10))

        self.weight_config_frame = tk.Frame(self)
        self.weight_config_frame.pack(pady=10)

        self.weight_config_frame.pack_forget()

        threshold_label = tk.Label(self.weight_config_frame, text="Threshold:")
        threshold_label.grid(row=0, column=0)
        self.threshold_spinbox = tk.Spinbox(self.weight_config_frame, from_=0, to=10000, increment=10, width=5)
        self.threshold_spinbox.grid(row=0, column=1)
        self.threshold_spinbox.delete(0, "end")
        self.threshold_spinbox.insert(0, "1000")  # Default value

        below_label = tk.Label(self.weight_config_frame, text="Weight for K_CORE < Threshold:")
        below_label.grid(row=1, column=0)
        self.below_spinbox = tk.Spinbox(self.weight_config_frame, from_=0, to=10, increment=0.1, format="%0.1f",
                                        width=5)
        self.below_spinbox.grid(row=1, column=1)
        self.below_spinbox.delete(0, "end")
        self.below_spinbox.insert(0, "1")  # Default value

        above_label = tk.Label(self.weight_config_frame, text="Weight for K_CORE >= Threshold:")
        above_label.grid(row=2, column=0)
        self.above_spinbox = tk.Spinbox(self.weight_config_frame, from_=0, to=10, increment=0.1, format="%0.1f",
                                        width=5)
        self.above_spinbox.grid(row=2, column=1)
        self.above_spinbox.delete(0, "end")
        self.above_spinbox.insert(0, "0.5")  # Default value

        self.apply_btn = tk.Button(self, text="Fit Curve", command=self.reprocess_file)
        self.apply_btn.pack(pady=10)

    def update_weight_visibility(self, *args):
        method = self.fitting_method.get()
        if method == 'Weighted':
            self.weight_config_frame.pack(pady=10)
        else:
            self.weight_config_frame.pack_forget()

    def show_format_table(self, event):
        format_win = tk.Toplevel(self)
        format_win.title("Format Requirements")

        lbl = tk.Label(format_win, text="Here is an example of the format requirement for the excel file.")
        lbl.pack(pady=10)

        table = tk.Frame(format_win)
        table.pack(pady=10)

        headers = ["DEPT", "BVI", "FFI", "T2LM", "K_CORE"]
        example_data = ["3800.0", "0.0321", "0.1817", "81.94", "1068.33"]
        rows = [headers, example_data, ["..."] * 5, ["..."] * 5, ["..."] * 5, ["..."] * 5]

        for i, row in enumerate(rows):
            for j, item in enumerate(row):
                cell = tk.Label(table, text=item, relief="solid", borderwidth=1, padx=5, pady=5)
                cell.grid(row=i, column=j, sticky="nsew")
                table.grid_columnconfigure(j, minsize=80)

        format_win.geometry(f"{80 * len(headers) + 40}x400")
        format_win.resizable(False, False)

        def on_leave(event):
            format_win.destroy()

        format_win.bind("<Leave>", on_leave)

    def coates_model(self, params, bvi, ffi):
        c, m, n = params
        return c * (bvi + ffi) ** m * (ffi / bvi) ** n

    def sdr_model(self, params, t2lm, bvi, ffi):
        a, i, j = params
        return a * t2lm ** i * (bvi + ffi) ** j

    def get_weighted_residual(self, predicted, k_core):
        threshold = float(self.threshold_spinbox.get())
        below_weight = float(self.below_spinbox.get())
        above_weight = float(self.above_spinbox.get())

        weights = np.where(k_core < threshold, below_weight, above_weight)
        return (k_core - predicted) * weights

    def residual_coates(self, params, bvi, ffi, k_core):
        method = self.fitting_method.get()
        predicted = self.coates_model(params, bvi, ffi)
        if method == 'Log Transformation':
            return np.log(k_core) - np.log(predicted)
        elif method == 'Weighted':
            return self.get_weighted_residual(predicted, k_core)
        elif method == 'Error Normalization':
            return (k_core - predicted) / k_core
        else:
            return k_core - predicted

    def residual_sdr(self, params, t2lm, bvi, ffi, k_core):
        method = self.fitting_method.get()
        predicted = self.sdr_model(params, t2lm, bvi, ffi)
        if method == 'Log Transformation':
            return np.log(k_core) - np.log(predicted)
        elif method == 'Weighted':
            return self.get_weighted_residual(predicted, k_core)
        elif method == 'Error Normalization':
            return (k_core - predicted) / k_core
        else:
            return k_core - predicted

    def display_equations(self, coates_params, sdr_params, container_frame):
        for widget in container_frame.winfo_children():
            widget.destroy()  # Clear previously displayed equations if any.

        fig, axs = plt.subplots(2, 1, figsize=(4, 2))

        coates_text = f"$K_{{Timur-Coates}} = {coates_params[0]:.2f} \cdot \Phi^{{{coates_params[1]:.2f}}} \cdot (\dfrac{{FFI}}{{BVI}})^{{{coates_params[2]:.2f}}}$"
        sdr_text = f"$K_{{SDR}} = {sdr_params[0]:.2f} \cdot {{T_{{2LM}}}}^{{{sdr_params[1]:.2f}}} \cdot \Phi^{{{sdr_params[2]:.2f}}}$"

        axs[0].text(0.5, 0.5, coates_text, size=12, ha='center', va='center')
        axs[1].text(0.5, 0.5, sdr_text, size=12, ha='center', va='center')

        for ax in axs:
            ax.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=container_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)

    def process_file(self):
        if self.current_file_path:
            self.current_file_path = self.current_file_path

        if not self.current_file_path:
            print("No file loaded yet.")
            return

        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, self.current_file_path)

        sheet_name = "Sheet1"

        try:
            df = pd.read_excel(self.current_file_path, sheet_name=sheet_name, engine="openpyxl")

            required_columns = ["BVI", "FFI", "T2LM", "K_CORE"]
            if not all(col in df.columns for col in required_columns):
                print("The Excel file doesn't have the required columns.")
                return

            filtered_df = df.dropna(subset=['K_CORE'])

            bvi_column = filtered_df["BVI"]
            ffi_column = filtered_df["FFI"]
            k_core_column = filtered_df["K_CORE"]
            t2lm_column = filtered_df["T2LM"]

        except Exception as e:
            print(f"Error reading Excel or extracting columns: {e}")
            return

        initial_guess_coates = [1.0, 4.0, 2.0]
        result_coates = least_squares(self.residual_coates, initial_guess_coates,
                                      args=(bvi_column, ffi_column, k_core_column))
        df["Calculated_K_Coates"] = self.coates_model(result_coates.x, df["BVI"], df["FFI"])

        initial_guess_sdr = [1.0, 2.0, 4.0]
        result_sdr = least_squares(self.residual_sdr, initial_guess_sdr,
                                   args=(t2lm_column, bvi_column, ffi_column, k_core_column))
        df["Calculated_K_SDR"] = self.sdr_model(result_sdr.x, df["T2LM"], df["BVI"], df["FFI"])

        if self.fitting_method.get() == 'Log Transformation':
            df["Calculated_K_Coates"] = np.exp(df["Calculated_K_Coates"])
            df["Calculated_K_SDR"] = np.exp(df["Calculated_K_SDR"])

        try:
            df.to_excel(self.current_file_path, sheet_name=sheet_name, index=False, engine="openpyxl")
        except Exception as e:
            print(f"Error writing back to Excel: {e}")
            return

        self.display_equations(result_coates.x, result_sdr.x, self.equations_frame)

    def browse_excel_file(self):
        file_path = filedialog.askopenfilename(title="Select Excel File",
                                               filetypes=[("Excel files", "*.xlsx;*.csv"), ("All files", "*.*")])
        if file_path:
            self.current_file_path = file_path  # Set the current file path
            self.process_file()

    def reprocess_file(self):
        if self.current_file_path:
            self.process_file()
        else:
            print("Please select a file first.")


if __name__ == "__main__":
    root = tk.Tk()
    app_frame = CMRPermeabilityApp(master=root)
    app_frame.pack(pady=20, padx=20)
    root.mainloop()
