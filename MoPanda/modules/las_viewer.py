import tkinter as tk
from tkinter import filedialog, messagebox
from EDA.MoPanda.MoPanda.modules.las_io import LasIO
import os
import pandas as pd

class WellLogGUI(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        # self.parent = parent

        self.title("LAS Viewer")
        self.geometry("800x500")

        self.las_file_path = '../data/las'
        self.output_path = '../output'
        self.log = None

        self.create_widgets()

    def create_widgets(self):
        self.select_file_button = tk.Button(self, text="Select LAS File", command=self.select_las_file)
        self.select_file_button.pack(pady=10)

        self.display_well_button = tk.Button(self, text="Display Well", command=self.display_well)
        self.display_well_button.pack(pady=5)

        self.display_params_button = tk.Button(self, text="Display Parameters", command=self.display_params)
        self.display_params_button.pack(pady=5)

        self.display_params_button = tk.Button(self, text="Display Curves", command=self.display_curves)
        self.display_params_button.pack(pady=5)

        self.select_output_button = tk.Button(self, text="Select Output Path", command=self.select_output_path)
        self.select_output_button.pack(pady=5)

        self.export_button = tk.Button(self, text="Export Data", command=self.export_data)
        self.export_button.pack(pady=5)

        # Text areas to display well information and drilling parameters
        self.text_well = tk.Text(self, width=140, height=50)
        self.text_well.pack(pady=5)

        # Label to display the selected output path
        self.output_label = tk.Label(self, text="Output Path: ")
        self.output_label.pack(pady=5)

    def select_las_file(self):
        self.las_file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if self.las_file_path:
            self.log = LasIO(self.las_file_path)
            messagebox.showinfo("Log Loaded", "Log loaded successfully.\nLAS file path: {}".format(self.las_file_path))

    def display_well(self):
        if self.log:
            well_info = str(self.log.well)
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, well_info)
            self.update()
        else:
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, "No LAS file loaded.")
            self.update()

    def display_params(self):
        if self.log:
            params_info = str(self.log.params)

            # Display the formatted string in the left text area
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, params_info)
            self.update()
        else:
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, "No LAS file loaded.")
            self.update()

    def display_curves(self):
        if self.log:
            curves_info = str(self.log.curves)

            # Display the formatted string in the right text area
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, curves_info)
            self.update()
        else:
            self.text_well.delete("1.0", tk.END)
            self.text_well.insert(tk.END, "No LAS file loaded.")
            self.update()

    def select_output_path(self):
        self.output_path = filedialog.askdirectory()
        self.output_label.config(text="Output Path: {}".format(self.output_path))
        self.update()

    def export_data(self):
        if self.log:
            if self.output_path:
                filename = os.path.splitext(os.path.basename(self.las_file_path))[0]
                csv_filename = filename + "_las2csv.csv"
                excel_filename = filename + "_las2excel.xlsx"
                csv_path = os.path.join(self.output_path, csv_filename)
                excel_path = os.path.join(self.output_path, excel_filename)

                self.log.export_csv(csv_path)
                self.log.export_excel(excel_path)
                messagebox.showinfo("Export Successful", "Data exported successfully.")
            else:
                messagebox.showerror("Export Error", "No output path selected.")
        else:
            messagebox.showerror("Export Error", "No LAS file loaded.")

if __name__ == "__main__":
    app = WellLogGUI()
    app.mainloop()
