import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import least_squares


class CMRPermeabilityApp(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky=(tk.N, tk.W, tk.E, tk.S))  # Embed it into the provided frame
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

    def residual_coates(self, params, bvi, ffi, k_core):
        return k_core - self.coates_model(params, bvi, ffi)

    def residual_sdr(self, params, t2lm, bvi, ffi, k_core):
        return k_core - self.sdr_model(params, t2lm, bvi, ffi)

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


    def process_file(self, excel_file_path):
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, excel_file_path)

        sheet_name = "Sheet1"

        try:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name, engine="openpyxl")

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
        result_coates = least_squares(self.residual_coates, initial_guess_coates, args=(bvi_column, ffi_column, k_core_column))
        df["Calculated_K_Coates"] = self.coates_model(result_coates.x, df["BVI"], df["FFI"])

        initial_guess_sdr = [1.0, 2.0, 4.0]
        result_sdr = least_squares(self.residual_sdr, initial_guess_sdr,
                                   args=(t2lm_column, bvi_column, ffi_column, k_core_column))
        df["Calculated_K_SDR"] = self.sdr_model(result_sdr.x, df["T2LM"], df["BVI"], df["FFI"])

        try:
            df.to_excel(excel_file_path, sheet_name=sheet_name, index=False, engine="openpyxl")
        except Exception as e:
            print(f"Error writing back to Excel: {e}")
            return

        self.display_equations(result_coates.x, result_sdr.x, self.equations_frame)


    def browse_excel_file(self):
        file_path = filedialog.askopenfilename(title="Select Excel File",
                                               filetypes=[("Excel files", "*.xlsx;*.csv"), ("All files", "*.*")])
        if file_path:
            self.process_file(file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app_frame = CMRPermeabilityApp(master=root)
    app_frame.pack(pady=20, padx=20)
    root.mainloop()
