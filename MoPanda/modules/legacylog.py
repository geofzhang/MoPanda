import PySimpleGUI as sg
import lasio
import pandas as pd
import numpy as np
import os


def load_lithology_correction_curves():
    curves = {}
    folder_path = "../data/chartbook/NEUT2NPHI/"

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


def neut_to_nphi(snl, lithology_correction_curve):
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

        self.functions = ["NEUT --> NPHI", "RHOB --> DPHI", "DPHI --> RHOB", "Function 4", "Function 5", "Function 6"]

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

        tools = {
            "SLB": ["GNAM 1-5", "GNT-F/G/H", "GNT-J/K", "GNT-N"],
            # Add tools for other service companies here
        }

        tool_info = {
            "GNAM 1-5": "Tool Type: GNAM 1-5\nSonde Diameter: 3 5/8\"\nFluid Filled Holes",
            # Add info for other tools here
        }

        while True:
            event, values = self.window.read()

            if event == sg.WIN_CLOSED:
                break

            if event == "-SERVICE_COMPANY-":
                service_company = values["-SERVICE_COMPANY-"]
                if service_company in tools:
                    self.window["-LBL_TOOL_TYPE-"].update(visible=True)
                    self.window["-TOOL_TYPE-"].update(values=tools[service_company], visible=True)
                else:
                    self.window["-LBL_TOOL_TYPE-"].update(visible=False)
                    self.window["-TOOL_TYPE-"].update(values=[], visible=False)
                    self.window["-TOOL_INFO-"].update(visible=False)

            if event == "-TOOL_TYPE-":
                tool_type = values["-TOOL_TYPE-"]
                if tool_type in tool_info:
                    self.window["-TOOL_INFO-"].update(tool_info[tool_type], visible=True)

            if event == "Load LAS File":
                self.file_path = values["-FILE-"]
                try:
                    self.las_file = lasio.read(self.file_path)
                    self.data_frame = self.las_file.df()
                    self.window["-FILEPATH-"].update(f"Loaded LAS file:\n{self.file_path}")
                except Exception as e:
                    self.window["-FILEPATH-"].update(f"Error loading LAS file:\n{str(e)}")
                    self.data_frame = None

            if event == "NEUT --> NPHI" and self.data_frame is not None:
                self.convert_neut_to_nphi(values)
            if event == "RHOB --> DPHI" and self.data_frame is not None:
                self.convert_rhob_to_dphi(values)
            if event == "DPHI --> RHOB" and self.data_frame is not None:
                self.convert_dphi_to_rhob(values)

        self.window.close()

    def convert_neut_to_nphi(self, values):
        lithology_correction_curves = load_lithology_correction_curves()

        # Let the user select the column for NEUT
        column_options = self.data_frame.columns.tolist()
        event, values = sg.Window("Log Selection", [
            [sg.Text("Select the log for conversion:")],
            [sg.InputCombo(column_options, size=(40, 1), key='-LOG_CHOICE-')],
            [sg.Button("OK"), sg.Button("Cancel")]
        ]).read(close=True)

        if event == "OK":
            self.log_choice = values['-LOG_CHOICE-']

        # Select a company
        company_options = ["SLB", "Halliburton", "Baker Hughes", "Weatherford", "Else"]
        event, values = sg.Window("Company Selection", [
            [sg.Text("Select a company:")],
            [sg.InputCombo(company_options, size=(40, 1), key='-COMPANY-')],
            [sg.Button("OK"), sg.Button("Cancel")]
        ]).read(close=True)

        if event == "OK" and values['-COMPANY-'] == "SLB":
            # Select a tool type within SLB
            slb_tool_options = ["GNAM 1-5", "GNT-F/G/H", "GNT-J/K", "GNT-N"]
            event, values = sg.Window("SLB Tool Selection", [
                [sg.Text("Select a tool type:")],
                [sg.InputCombo(slb_tool_options, size=(40, 1), key='-SLB_TOOL-')],
                [sg.Button("OK"), sg.Button("Cancel")]
            ]).read(close=True)

            if event == "OK" and values['-SLB_TOOL-'] == "GNT-J/K":
                # Lithology options under GNT-J/K
                lithology_options = ["Sandstone", "Limestone", "Dolomite"]
                event, values = sg.Window("Lithology Correction", [
                    [sg.Text("Select a lithology correction:")],
                    [sg.InputCombo(lithology_options, size=(40, 1), key='-LITHOLOGY-')],
                    [sg.Button("OK"), sg.Button("Cancel")]
                ]).read(close=True)

                if event == "OK":
                    selected_lithology = values['-LITHOLOGY-']
                    lithology_correction_curve = lithology_correction_curves.get(selected_lithology)
                    if lithology_correction_curve is not None:
                        corrected_nphi = neut_to_nphi(self.data_frame[self.log_choice],
                                                      lithology_correction_curve)
                        self.las_file["NPHI_N"] = corrected_nphi
                        self.las_file.write(self.file_path, version=2)
                        self.window["-FILEPATH-"].update(
                            f"Data converted and written to LAS file:\n{self.file_path}")
                    else:
                        sg.popup_error(f"Lithology correction curve for {selected_lithology} not found.")

            if event == "OK" and values['-SLB_TOOL-'] == "GNAM 1-5":
                # Combine Porosity Index Information, Cased Hole Checkbox, and Borehole Diameter Selection in one window
                borehole_options = ["4 3/4\"", "6\"", "6 1/2\"", "7\"", "8\"", "9\"", "10\"",
                                    "Manual Input (Max NEUT Reading)"]
                diameters_to_x = {"4 3/4\"": 850, "6\"": 648, "6 1/2\"": 592, "7\"": 533, "8\"": 477,
                                  "9\"": 417, "10\"": 380}

                layout = [
                    [sg.Text("Porosity Index (GNAM 1-5) Information")],
                    [sg.Text("Tool Type: GNAM 1-5\nSonde Diameter: 3 5/8\"\nFluid Filled Holes")],
                    [sg.Text("Is this a cased hole?")],
                    [sg.Checkbox("", default=False, key='-CASED-')],
                    [sg.Text("Select or Input Borehole Diameter:")],
                    [sg.Combo(borehole_options, key='-BOREHOLE-', enable_events=True)],
                    [sg.InputText("", key='-MANUAL_INPUT-', visible=False)],
                    [sg.Button("OK"), sg.Button("Cancel")]
                ]

                window = sg.Window("GNAM 1-5 Settings", layout)

                borehole_choice = None
                x = 0

                while True:
                    event, values = window.read()
                    if event == sg.WIN_CLOSED or event == "Cancel":
                        break

                    if event == '-BOREHOLE-':  # If borehole dropdown is changed
                        if values['-BOREHOLE-'] == "Manual Input (Max NEUT Reading)":
                            window['-MANUAL_INPUT-'].update(visible=True)
                        else:
                            window['-MANUAL_INPUT-'].update(visible=False)
                            borehole_choice = values['-BOREHOLE-']
                            x = diameters_to_x.get(borehole_choice, 0)

                    if event == "OK":
                        if values['-BOREHOLE-'] == "Manual Input (Max NEUT Reading)":
                            x = float(values['-MANUAL_INPUT-'])

                        if x > 0:
                            # Calculate Y based on the linear function
                            y = ((0 - 1.662) / (x - 170)) * (self.data_frame[self.log_choice] - 170) + 1.662
                            # Convert Y to NPHI
                            nphi = np.power(10, y) / 100
                            # Write the converted NPHI back to the LAS file
                            self.las_file["NPHI_N"] = nphi
                            self.las_file.write(self.file_path, version=2)

                            self.window["-FILEPATH-"].update(
                                f"Data converted and written to LAS file:\n{self.file_path}")
                        break  # Close window after clicking OK

                window.close()
        pass

    def convert_rhob_to_dphi(self, values):
        column_options = self.data_frame.columns.tolist()
        scale_options = ["Limestone", "Sandstone", "Dolomite"]
        scales = {"Limestone": 2.71, "Sandstone": 2.65, "Dolomite": 2.87}

        layout = [
            [sg.Text("Select the log for RHOB conversion:")],
            [sg.Combo(column_options, key='-RHOB_LOG-', size=(40, 1))],
            [sg.Text("Select the scale for DPHI calculation:")],
            [sg.Combo(scale_options, key='-SCALE-', size=(40, 1))],
            [sg.Button("Calculate"), sg.Button("Cancel")]
        ]

        window = sg.Window("RHOB to DPHI Conversion", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            if event == "Calculate":
                rhob_log = values['-RHOB_LOG-']
                scale = scales[values['-SCALE-']]
                dphi = (scale - self.data_frame[rhob_log]) / (scale - 1)
                # Assuming 'las_file' is a global or class-level variable
                self.las_file["DPHI_N"] = dphi
                self.las_file.write(self.file_path, version=2)  # Ensure 'file_path' is accessible here
                self.window["-FILEPATH-"].update(
                    f"DPHI data calculated and written to LAS file:\n{self.file_path}")
                break

        window.close()

    def convert_dphi_to_rhob(self, values):
        column_options = self.data_frame.columns.tolist()
        scale_options = ["Limestone", "Sandstone", "Dolomite"]
        scales = {"Limestone": 2.71, "Sandstone": 2.65, "Dolomite": 2.87}

        layout = [
            [sg.Text("Select the log for DPHI conversion:")],
            [sg.Combo(column_options, key='-DPHI_LOG-', size=(40, 1))],
            [sg.Text("Select the scale for RHOB calculation:")],
            [sg.Combo(scale_options, key='-SCALE-', size=(40, 1))],
            [sg.Button("Calculate"), sg.Button("Cancel")]
        ]

        window = sg.Window("DPHI to RHOB Conversion", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            if event == "Calculate":
                dphi_log = values['-DPHI_LOG-']
                scale = scales[values['-SCALE-']]
                rhob = scale - (scale - 1) * self.data_frame[dphi_log]
                # Assuming 'las_file' is a global or class-level variable
                self.las_file["RHOB_N"] = rhob
                self. las_file.write(self.file_path, version=2)  # Ensure 'file_path' is accessible here
                self.window["-FILEPATH-"].update(
                    f"RHOB data calculated and written to LAS file:\n{self.file_path}")
                break

        window.close()


if __name__ == "__main__":
    app = LegacyLogConverter()
    app.run()
