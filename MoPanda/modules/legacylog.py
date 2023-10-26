import PySimpleGUI as sg
import lasio
import pandas as pd
import numpy as np
import os


def load_lithology_correction_curves():
    curves = {}
    folder_path = "../data/chartbook/SNL2NPHI/"

    if not os.path.exists(folder_path):
        sg.popup_error(f"Curve folder '{folder_path}' not found. Please check the folder location.")
        return None

    for lithology in ["Sandstone", "Limestone", "Dolomite"]:
        curve_path = os.path.join(folder_path, f"{lithology}.csv")
        if os.path.exists(curve_path):
            curve_data = pd.read_csv(curve_path, header=None, names=["X", "Y"])
            curves[lithology] = np.array(curve_data)
        else:
            sg.popup_error(f"Curve file '{lithology}.csv' not found. Please check the file location.")

    return curves


def snl_to_nphi(snl, lithology_correction_curve):
    x_values = lithology_correction_curve[:, 1][::-1]  # Reverse the order of X values
    y_values = lithology_correction_curve[:, 0][::-1]  # Reverse the order of Y values
    converted_column = np.interp(snl, x_values, y_values)
    return converted_column


class LegacyLogConverter:
    def __init__(self):

        self.data_frame = None

        sg.theme("Default1")

        self.layout = [
            [sg.Text("Legacy Log Converter", font="Any 12 bold")],
            [sg.InputText("", key="-FILE-", size=(60, 2)),
             sg.FileBrowse(size=(10, 1), file_types=(("LAS Files", "*.las"), ("All Files", "*.*")))],
            [sg.Button("Load LAS File", size=(20, 1))],
            [sg.Multiline("", key="-FILEPATH-", size=(72, 5), disabled=True)],
            [sg.Text("Log Conversions", justification="left")],
        ]

        self.functions = ["SNL --> NPHI", "Function 2", "Function 3", "Function 4", "Function 5", "Function 6"]

        self.button_layout = []

        for i in range(0, len(self.functions), 3):
            row = []
            for j in range(3):
                if i + j < len(self.functions):
                    button = sg.Button(self.functions[i + j], size=(20, 1))
                    row.append(button)
            self.button_layout.append(row)

        for row in self.button_layout:
            self.layout.append(row)

        self.window = sg.Window("Individual Log Calculator", self.layout, size=(550, 600))

    def run(self):
        lithology_correction_curves = load_lithology_correction_curves()

        while True:
            event, values = self.window.read()

            if event == sg.WIN_CLOSED:
                break

            if event == "Load LAS File":
                file_path = values["-FILE-"]
                try:
                    las_file = lasio.read(file_path)
                    self.data_frame = las_file.df()
                    self.window["-FILEPATH-"].update(f"Loaded LAS file:\n{file_path}")
                except Exception as e:
                    self.window["-FILEPATH-"].update(f"Error loading LAS file:\n{str(e)}")
                    self.data_frame = None

            if self.data_frame is not None and event == "SNL --> NPHI":
                log_names = self.data_frame.columns.tolist()
                event, values = sg.Window("Log Selection", [
                    [sg.Text("Select a Log for conversion:")],
                    [sg.InputCombo(log_names, size=(40, 1), key='-LOG-')],
                    [sg.Button("OK"), sg.Button("Cancel")]
                ]).read(close=True)

                if event == "OK":
                    column_choice = values['-LOG-']

                    lithology_options = ["Sandstone", "Limestone", "Dolomite"]
                    event, values = sg.Window("Lithology Correction", [
                        [sg.Text("Select a lithology correction:")],
                        [sg.InputCombo(lithology_options, size=(40, 1), key='-LITHOLOGY-')],
                        [sg.Button("OK"), sg.Button("Cancel")]
                    ]).read(close=True)

                    if event == "OK":
                        lithology_choice = values['-LITHOLOGY-']
                        if lithology_choice in lithology_correction_curves:
                            lithology_correction_curve = lithology_correction_curves[lithology_choice]
                            converted_data = snl_to_nphi(self.data_frame[column_choice], lithology_correction_curve)
                            las_file = lasio.read(file_path)
                            las_file["NPHI_N"] = converted_data
                            las_file.write(file_path, version=2)

                            self.window["-FILEPATH-"].update(f"Data converted and written to LAS file:\n{file_path}")

        self.window.close()


if __name__ == "__main__":
    app = LegacyLogConverter()
    app.run()
