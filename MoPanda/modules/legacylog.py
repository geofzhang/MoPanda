import PySimpleGUI as sg
import lasio
import pandas as pd
import numpy as np
import os
import datetime


def load_lithology_correction_curves():
    curves = {}
    folder_path = "./data/chartbook/NEUT2NPHI/"

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
        self.scales = {"Limestone": 2.71, "Sandstone": 2.65, "Dolomite": 2.87}
        self.scale_options = ["Limestone", "Sandstone", "Dolomite"]
        self.offset = {
            "SNP (Sidewall Neutron)": {
                ("Sandstone", "Limestone"): -0.03,
                ("Sandstone", "Dolomite"): -0.06,
                ("Limestone", "Sandstone"): 0.03,
                ("Limestone", "Dolomite"): -0.03,
                ("Dolomite", "Sandstone"): 0.06,
                ("Dolomite", "Limestone"): 0.03
            },
            "CNL (Compensated Neutron)": {
                ("Sandstone", "Limestone"): -0.04,
                ("Sandstone", "Dolomite"): -0.07,
                ("Limestone", "Sandstone"): 0.04,
                ("Limestone", "Dolomite"): -0.03,
                ("Dolomite", "Sandstone"): 0.07,
                ("Dolomite", "Limestone"): 0.03
            }
        }
        sg.theme("Default1")

        self.layout = [
            [sg.Text("Legacy Log Converter", font="Any 10 bold")],
            [sg.InputText("", key="-FILE-", size=(60, 2)),
             sg.FileBrowse(size=(10, 1), file_types=(("LAS Files", "*.las"), ("All Files", "*.*")))],
            [sg.Button("Load LAS File", size=(20, 1))],
            [sg.Multiline("", key="-FILEPATH-", size=(72, 5), disabled=True)],
            [sg.Text("Log Conversions", justification="left")],
        ]

        self.functions = ["NEUT --> NPHI", "RHOB --> DPHI", "DPHI --> RHOB", "% --> Fraction", "Lithology Correction",
                          "Function 6"]

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
            if event == "% --> Fraction" and self.data_frame is not None:
                self.convert_perc_to_frac()
            if event == "Lithology Correction" and self.data_frame is not None:
                self.lithology_correction()

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
                        self.las_file.curves["NPHI_N"].unit = "v/v"
                        self.las_file.curves["NPHI_N"].descr = f"Neutron porosity (From SLB GNT-J/K)"
                        self.output()
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
                            self.las_file.curves["NPHI_N"].unit = "v/v"
                            self.las_file.curves["NPHI_N"].descr = f"Neutron porosity (From SLB GNAM 1-5)"
                            self.output()
                        break  # Close window after clicking OK

                window.close()
        pass

    def convert_rhob_to_dphi(self, values):
        column_options = self.data_frame.columns.tolist()

        layout = [
            [sg.Text("Select the log for RHOB conversion:")],
            [sg.Combo(column_options, key='-RHOB_LOG-', size=(40, 1))],
            [sg.Text("Select the scale for DPHI calculation:")],
            [sg.Combo(self.scale_options, key='-SCALE-', size=(40, 1))],
            [sg.Button("Calculate"), sg.Button("Cancel")]
        ]

        window = sg.Window("RHOB to DPHI Conversion", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            if event == "Calculate":
                rhob_log = values['-RHOB_LOG-']
                scale = self.scales[values['-SCALE-']]
                dphi = (scale - self.data_frame[rhob_log]) / (scale - 1)
                # Assuming 'las_file' is a global or class-level variable
                self.las_file["DPHI_N"] = dphi
                self.las_file.curves["DPHI_N"].unit = "v/v"
                self.las_file.curves["DPHI_N"].descr = f"Density porosity ({scale} scale)"
                self.output()
                break

        window.close()

    def convert_dphi_to_rhob(self, values):
        column_options = self.data_frame.columns.tolist()

        layout = [
            [sg.Text("Select the log for DPHI conversion:")],
            [sg.Combo(column_options, key='-DPHI_LOG-', size=(40, 1))],
            [sg.Text("Select the scale for RHOB calculation:")],
            [sg.Combo(self.scale_options, key='-SCALE-', size=(40, 1))],
            [sg.Button("Calculate"), sg.Button("Cancel")]
        ]

        window = sg.Window("DPHI to RHOB Conversion", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            if event == "Calculate":
                dphi_log = values['-DPHI_LOG-']
                scale = self.scales[values['-SCALE-']]
                rhob = scale - (scale - 1) * self.data_frame[dphi_log]
                # Assuming 'las_file' is a global or class-level variable
                self.las_file["RHOB_N"] = rhob
                self.las_file.curves["RHOB_N"].unit = "g/cc"
                self.las_file.curves["RHOB_N"].descr = f"Bulk density ({scale} scale)"
                self.output()
                break

        window.close()

    def convert_perc_to_frac(self):
        column_options = self.data_frame.columns.tolist()
        layout = [
            [sg.Text("Select Porosity logs for unit conversion:")],
            [sg.Listbox(column_options, key='-POROSITY_LOG-', size=(40, 6),
                        select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE)],
            [sg.Button("Convert"), sg.Button("Cancel")]
        ]
        window = sg.Window("Percentage to Fraction Conversion", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            if event == "Convert":
                selected_logs = values['-POROSITY_LOG-']
                for log in selected_logs:
                    phi_frac = self.data_frame[log] / 100
                    self.data_frame[log] = phi_frac
                    self.las_file.curves[log].unit = 'v/v'
                    self.las_file.curves[log].data = phi_frac
                    self.las_file.curves[log].descr = f'{log} (Unit converted to fraction)'

                self.output()
                break

        window.close()

    def lithology_correction(self):
        column_options = self.data_frame.columns.tolist()
        rigorous_options = [option for option in self.scale_options if option != "Limestone"]
        operator_list = ["Schlumberger", "Dresser Atlas", "Welex/Halliburton"]

        # Tab for Density Porosity
        density_tab_layout = [
            [sg.Text("Select Density Porosity log:")],
            [sg.Combo(column_options, key='-DENSITY_POROSITY_LOG-', size=(40, 1), enable_events=True)],
            [sg.Text("From:"), sg.Combo(self.scale_options, key='-FROM_LITHOLOGY-', size=(10, 1), enable_events=True),
             sg.Text("To:"), sg.Combo(self.scale_options, key='-TO_LITHOLOGY-', size=(10, 1), enable_events=True)]
        ]

        # Sub-tabs for Neutron Porosity Tab
        generic_methods_layout = [
            [sg.Text("Logging Tool:"), sg.Combo(["SNP (Sidewall Neutron)", "CNL (Compensated Neutron)"],
                                                key='-LOGGING_TOOL_N-', size=(20, 1), enable_events=True)],
            [sg.Text("From:"), sg.Combo(self.scale_options, key='-FROM_LITHOLOGY_N-', size=(10, 1), enable_events=True),
             sg.Text("To:"), sg.Combo(self.scale_options, key='-TO_LITHOLOGY_N-', size=(10, 1), enable_events=True)]
        ]

        # Rigorous Methods Sub-Tab
        rigorous_methods_layout = [
            [sg.Text("WARNING: \n"
                     "Only corrects Limestone scale to others\n"
                     "Use only when tool and operator are known",
                     text_color='orange', font=("Helvetica", 9))],
            [sg.Text("Logging Tool:"), sg.Combo(["SNP (Sidewall Neutron)", "CNL (Compensated Neutron)"],
                                                key='-LOGGING_TOOL_R-', size=(20, 1), enable_events=True)],
            [sg.Text("Service Company:"), sg.Combo(operator_list,
                                                   key='-SERVICE_COMPANY-', size=(20, 1))],
            [sg.Text("From: Limestone"), sg.Text("To:"),
             sg.Combo(rigorous_options, key='-TO_LITHOLOGY_R-', size=(10, 1))]
        ]

        # Layout for Neutron Porosity Tab
        neutron_tab_layout = [
            [sg.Text("Select Neutron Porosity log:")],
            [sg.Combo(column_options, key='-NEUTRON_POROSITY_LOG-', size=(40, 1), enable_events=True)],
            [sg.TabGroup([[sg.Tab('Generic Methods', generic_methods_layout, key='-GENERIC_METHODS-'),
                           sg.Tab('Rigorous Methods', rigorous_methods_layout, key='-RIGOROUS_METHODS-')]],
                         key='-SUB_TAB-')]
        ]

        # Tab group
        tab_group_layout = [[sg.Tab('Density Porosity', density_tab_layout, key='-DPHI-'),
                             sg.Tab('Neutron Porosity', neutron_tab_layout, key='-NPHI-')]
                            ]

        layout = [
            [sg.TabGroup(tab_group_layout, key='-TAB_GROUP-')],
            [sg.Button("Correct"), sg.Button("Cancel")],
            [sg.Multiline("", key="-LOG_INFO-", size=(40, 5), disabled=True)]
        ]

        window = sg.Window("Lithology Correction of Porosity", layout)

        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, "Cancel"):
                break

            # Handling log selection in both main tab and sub-tabs
            if event in ('-DENSITY_POROSITY_LOG-', '-NEUTRON_POROSITY_LOG-'):
                selected_log = values[event]
                if selected_log:
                    mnemonic = selected_log
                    unit = self.las_file.curves[selected_log].unit
                    description = self.las_file.curves[selected_log].descr
                    window["-LOG_INFO-"].update(f"Selected Log: {mnemonic}\nUnit: {unit}\nDescription: {description}\n")

            # Handling lithology selection for both Density and Neutron Porosity
            if event in ('-FROM_LITHOLOGY-', '-TO_LITHOLOGY-', '-FROM_LITHOLOGY_N-', '-TO_LITHOLOGY_N-'):
                opposite_key_mappings = {
                    '-FROM_LITHOLOGY-': '-TO_LITHOLOGY-',
                    '-TO_LITHOLOGY-': '-FROM_LITHOLOGY-',
                    '-FROM_LITHOLOGY_N-': '-TO_LITHOLOGY_N-',
                    '-TO_LITHOLOGY_N-': '-FROM_LITHOLOGY_N-'
                }
                selected_lithology = values[event]
                opposite_key = opposite_key_mappings[event]
                current_opposite_selection = values[opposite_key]
                opposite_options = [l for l in self.scale_options if
                                    l != selected_lithology or l == current_opposite_selection]

                window[opposite_key].update(values=opposite_options, set_to_index=opposite_options.index(
                    current_opposite_selection) if current_opposite_selection in opposite_options else 0)

            # Update service company options based on selected logging tool
            if event == '-LOGGING_TOOL_R-':
                selected_tool = values['-LOGGING_TOOL_R-']
                if selected_tool == "SNP (Sidewall Neutron)":
                    window['-SERVICE_COMPANY-'].update(values=["Schlumberger", "Dresser Atlas"])
                else:
                    window['-SERVICE_COMPANY-'].update(values=operator_list)

            if event == "Correct":
                selected_tab_title = window['-TAB_GROUP-'].get()
                if selected_tab_title == '-DPHI-':
                    from_lithology = values['-FROM_LITHOLOGY-']
                    to_lithology = values['-TO_LITHOLOGY-']
                    selected_log = values['-DENSITY_POROSITY_LOG-']

                    if from_lithology and to_lithology and selected_log:
                        # Calculate new density porosity
                        dphi = self.data_frame[selected_log]
                        rhob = self.scales[from_lithology] - dphi * (self.scales[from_lithology] - 1)
                        dphi_new = (self.scales[to_lithology] - rhob) / (self.scales[to_lithology] - 1)

                        # Update the data frame with new values
                        self.data_frame[selected_log + "_new"] = dphi_new
                        # Assuming 'las_file' is a global or class-level variable
                        self.las_file["DPHI_N"] = dphi_new
                        self.las_file.curves["DPHI_N"].unit = "v/v"
                        self.las_file.curves[
                            "DPHI_N"].descr = f"Lithology Corrected Density porosity ({to_lithology} scale)"

                if selected_tab_title == '-NPHI-':
                    selected_tab = values['-SUB_TAB-']
                    selected_log = values['-NEUTRON_POROSITY_LOG-']

                    if selected_tab == '-GENERIC_METHODS-':
                        to_lithology = values['-TO_LITHOLOGY_N-']
                        from_lithology = values['-FROM_LITHOLOGY_N-']
                        logging_tool = values['-LOGGING_TOOL_N-']

                        if selected_log and from_lithology and to_lithology and logging_tool:
                            NPHI = self.data_frame[selected_log]
                            offset = self.offset.get(logging_tool, {}).get((from_lithology, to_lithology))
                            if offset is not None:
                                NPHI_corrected = NPHI + offset
                                # Update the DataFrame and LAS file with NPHI_corrected
                                new_col_name = f"NPHI_{to_lithology[:2].upper()}"  # e.g., "NPHI_SS" for Sandstone

                                self.data_frame[new_col_name] = NPHI_corrected
                                self.las_file['NPHI_N'] = NPHI_corrected


                    elif selected_tab == '-RIGOROUS_METHODS-':
                        # Logic for rigorous methods
                        to_lithology = values['-TO_LITHOLOGY_R-']
                        logging_tool = values['-LOGGING_TOOL_R-']
                        service_company = values['-SERVICE_COMPANY-']
                        if selected_log and to_lithology and logging_tool and service_company:
                            NPHI_LS = self.data_frame[selected_log]  # Limestone porosity

                            # Calculate new porosity based on the selected options
                            if logging_tool == "SNP (Sidewall Neutron)":
                                if to_lithology == "Sandstone":
                                    if service_company == "Schlumberger":
                                        NPHI_SS = 0.222 * NPHI_LS ** 2 + 1.021 * NPHI_LS + 0.024
                                    elif service_company == "Dresser Atlas":
                                        NPHI_SS = -0.14 * NPHI_LS ** 2 + 1.047 * NPHI_LS + 0.0482
                                    # Update data frame with new values
                                    self.data_frame[selected_log + "_SS"] = NPHI_SS
                                    self.las_file["NPHI_N"] = NPHI_SS

                                elif to_lithology == "Dolomite":
                                    if service_company == "Schlumberger":
                                        NPHI_DL = 0.60 * NPHI_LS ** 2 + 0.7490 * NPHI_LS - 0.00434
                                    elif service_company == "Dresser Atlas":
                                        NPHI_DL = 0.34 * NPHI_LS ** 2 + 0.8278 * NPHI_LS - 0.01249
                                    # Update data frame with new values
                                    self.data_frame[selected_log + "_DL"] = NPHI_DL
                                    self.las_file["NPHI_N"] = NPHI_DL

                            elif logging_tool == "CNL (Compensated Neutron)":
                                if to_lithology == "Sandstone":
                                    if service_company == "Schlumberger":
                                        NPHI_SS = 0.222 * NPHI_LS ** 2 + 1.021 * NPHI_LS + 0.039
                                    elif service_company == "Dresser Atlas":
                                        NPHI_SS = NPHI_LS + 0.04
                                    elif service_company == "Welex/Halliburton":
                                        NPHI_SS = - 0.4778 * NPHI_LS ** 2 + 1.220 * NPHI_LS + 0.0311
                                    # Update data frame with new values
                                    self.data_frame[selected_log + "_SS"] = NPHI_SS
                                    self.las_file["NPHI_N"] = NPHI_SS

                                elif to_lithology == "Dolomite":
                                    if service_company == "Schlumberger":
                                        NPHI_DL = 1.40 * NPHI_LS ** 2 + 0.389 * NPHI_LS - 0.01259
                                    elif service_company == "Dresser Atlas":
                                        NPHI_DL_initial = 3.11 * NPHI_LS ** 2 + 0.102 * NPHI_LS - 0.00133

                                        # Conditional logic based on NPHI_DL_initial
                                        if NPHI_DL_initial < 0.10:
                                            NPHI_DL = NPHI_DL_initial
                                        else:
                                            NPHI_DL = NPHI_LS - 0.06
                                    elif service_company == "Welex/Halliburton":
                                        NPHI_DL = 1.397 * NPHI_LS ** 2 + 0.345 * NPHI_LS - 0.0152
                                    # Update data frame with new values
                                    self.data_frame[selected_log + "_DL"] = NPHI_DL
                                    self.las_file["NPHI_N"] = NPHI_DL

                    self.las_file.curves["NPHI_N"].unit = "v/v"
                    self.las_file.curves[
                        "NPHI_N"].descr = f"Lithology Corrected Neutron porosity ({to_lithology} scale)"

                # Output porosity logs with lithology correction
                self.output()
                break

        window.close()

    def output(self):
        output_dir = "./output/Conversion/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_file_path = os.path.join(output_dir, f"{original_filename}_conversion_{timestamp}.las")

        try:
            self.las_file.write(new_file_path, version=2)
            self.window["-FILEPATH-"].update(
                f"Data converted and written to new LAS file:\n{new_file_path}")
        except Exception as e:
            self.window["-FILEPATH-"].update(
                f"Error saving LAS file:\n{str(e)}")


if __name__ == "__main__":
    app = LegacyLogConverter()
    app.run()
