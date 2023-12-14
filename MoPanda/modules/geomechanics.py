import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
import lasio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GeoMechanics:
    def __init__(self, master):
        self.pp = None
        self.pext = 0  # Unbalanced Tectonic Stress in psi
        self.nct = None
        self.depth = None
        self.rhob = None
        self.dts = None
        self.dtc = None
        self.dtcw = 203.2
        self.dataframe = None
        self.file_path = None
        # Define units and descriptions for each curve
        self.curve_attributes = {
            'VRATIO': {'unit': '', 'desc': 'Dynamic Vp/Vs ratio'},
            'E_DYN': {'unit': 'psi', 'desc': 'Dynamic Young’s Modulus'},
            'L_DYN': {'unit': 'psi', 'desc': 'Dynamic 1st Lame Parameter'},
            'G_DYN': {'unit': 'psi', 'desc': 'Dynamic Shear Modulus'},
            'K_DYN': {'unit': 'psi', 'desc': 'Dynamic Bulk Modulus'},
            'NU_DYN': {'unit': '', 'desc': 'Dynamic Poisson’s Ratio'},
            'POB_DYN': {'unit': 'psi', 'desc': 'Dynamic Overburden Pressure'},
            'POBG_DYN': {'unit': 'psi/ft', 'desc': 'Dynamic Overburden Pressure Gradient'},
            'PP_DYN': {'unit': 'psi', 'desc': 'Dynamic Pore Pressure'},
            'PPG_DYN': {'unit': 'psi/ft', 'desc': 'Dynamic Pore Pressure Gradient'},
            'BIOT_DYN': {'unit': '', 'desc': "Dynamic Biots coefficient"},
            'MINHS_DYN': {'unit': 'psi', 'desc': 'Dynamic Minimum Horizontal Pressure(Closure Pressure)'},
            'MINHS90_DYN': {'unit': 'psi', 'desc': 'Dynamic Closure Pressure(90%)'},
            'INJECTABLE_DYN': {'unit': 'psi', 'desc': 'Dynamic Injectable Pressure'},
            'PFRAC_DYN': {'unit': 'psi', 'desc': 'Dynamic Fracture Pressure'},
            'FG_DYN': {'unit': 'psi/ft', 'desc': "Dynamic Fracture Gradient"},
        }

        # Create output directory
        output_dir = "./output/Geomechanics"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

        self.root = tk.Toplevel(master)
        self.root.title("Geo-Mechanics")

        tk.Label(self.root, text="Loading Functions", font="Helvetica 10 bold").grid(row=0, column=0, pady=10,
                                                                                     sticky="w")

        # Entry widget to display the selected file
        self.selected_file_entry = tk.Entry(self.root, width=80)
        self.selected_file_entry.grid(row=1, column=0, columnspan=3, padx=5, pady=0)
        self.selected_file_entry.bind("<FocusOut>", self.update_filename)
        self.selected_file_entry.bind("<Return>", self.update_filename)

        self.upload_button = tk.Button(self.root, width=20, text="Load Input File", command=self.upload_file)
        self.upload_button.grid(row=1, column=3, padx=5, pady=0)

        self.auto_load_button = tk.Button(self.root, width=20, text="Auto Load Logs", command=self.auto_load_logs)
        self.auto_load_button.grid(row=2, column=0, padx=5, pady=0)

        self.manual_load_button = tk.Button(self.root, width=20, text="Manual Load Logs", command=self.manual_load_logs)
        self.manual_load_button.grid(row=2, column=1, padx=5, pady=0)

        tk.Label(self.root, text="Dynamic Geo-mechanics", font="Helvetica 10 bold").grid(row=3, column=0, pady=10,
                                                                                         sticky="w")

        self.calculate_moduli_button = tk.Button(self.root, width=20, text="Elastic Modulus",
                                                 command=self.calculate_moduli)
        self.calculate_moduli_button.grid(row=4, column=0, padx=5)

        self.calculate_pob_button = tk.Button(self.root, width=20, text="Overburden Pressure",
                                              command=self.open_overburden_pressure)
        self.calculate_pob_button.grid(row=5, column=0, padx=5)

        self.calculate_pp_button = tk.Button(self.root, width=20, text="Pore Pressure", command=self.open_pore_pressure)
        self.calculate_pp_button.grid(row=5, column=1, padx=5)

        self.calculate_biot_button = tk.Button(self.root, width=20, text="Biot's Coefficient", command=self.open_biot)
        self.calculate_biot_button.grid(row=5, column=2, padx=5)

        self.calculate_min_hstress_button = tk.Button(self.root, width=45, text="Closure Stress and Frac Pressure",
                                                      command=self.calculate_min_hstress)
        self.calculate_min_hstress_button.grid(row=6, column=0, columnspan=2, padx=5)

        self.save_dynamic_button = tk.Button(self.root, width=45, text="Output Dynamic Geomechanical "
                                                                       "Properties",
                                             command=self.save_dynamic_results)
        self.save_dynamic_button.grid(row=7, column=0, columnspan=2, pady=5, padx=5)

        tk.Label(self.root, text="Static Geo-mechanics", font="Helvetica 10 bold").grid(row=8, column=0, pady=10,
                                                                                        sticky="w")

        self.progress_text = tk.Text(self.root, height=5, width=81)
        self.progress_text.grid(row=9, column=0, columnspan=4, pady=5, padx=0)

        self.root.mainloop()

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("All supported files", "*.csv, *.xlsx, *.las"),
                                                               ("CSV files", "*.csv"),
                                                               ("Excel files", "*.xlsx"),
                                                               ("LAS files", "*.las")])
        if self.file_path:
            try:
                if self.file_path.lower().endswith('.csv'):
                    self.dataframe = pd.read_csv(self.file_path, index_col=0)
                elif self.file_path.lower().endswith('.xlsx'):
                    if 'Curves' not in pd.ExcelFile(self.file_path).sheet_names:
                        messagebox.showerror("Error", "The .xlsx file does not have a 'Curves' sheet.")
                        return
                    self.dataframe = pd.read_excel(self.file_path, sheet_name='Curves', index_col=0)
                elif self.file_path.lower().endswith('.las'):
                    las = lasio.read(self.file_path)
                    self.dataframe = las.df().reset_index()
                # Update the selected file entry with the file path
                self.selected_file_entry.delete(0, tk.END)
                self.selected_file_entry.insert(0, self.file_path)
                self.progress_text.delete(1.0, tk.END)
                self.progress_text.insert(tk.END, "DataFrame loaded successfully!\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_filename(self, event=None):
        # Logic to update the filename
        self.file_path = self.selected_file_entry.get()

    def auto_load_logs(self):
        if self.dataframe is None:
            messagebox.showerror("Error", "No dataframe loaded.")
            return
        try:
            self.depth = self.dataframe['DEPT']
            self.dtc = self.dataframe['DTC_N']
            self.dts = self.dataframe['DTS_N']
            self.rhob = self.dataframe['RHOB_N']
            self.progress_text.insert(tk.END, "Required logs for dynamic geomechanics calculation loaded\n")
        except KeyError:
            messagebox.showerror("Error", "Required logs: DTC_N/ DTS_N/ RHOB_N not found in file.")

    def manual_load_logs(self):
        if self.dataframe is None:
            messagebox.showerror("Error", "No dataframe loaded.")
            return

        self.manual_selection_window = tk.Toplevel(self.root)
        self.manual_selection_window.title("Manual Log Selection")

        # Search bars and listboxes for each log
        self.create_log_selector("DTC", 0)
        self.create_log_selector("DTS", 1)
        self.create_log_selector("RHOB", 2)

        # Confirm button
        confirm_button = tk.Button(self.manual_selection_window, text="Confirm Selection",
                                   command=self.confirm_log_selection)
        confirm_button.grid(row=3, column=0, columnspan=2)

    def create_log_selector(self, log_name, row):
        tk.Label(self.manual_selection_window, text=f"Search for {log_name}").grid(row=row, column=0)
        search_var = tk.StringVar()
        search_var.trace("w", lambda name, index, mode, sv=search_var: self.update_listbox(log_name, sv))
        search_entry = tk.Entry(self.manual_selection_window, textvariable=search_var)
        search_entry.grid(row=row, column=1)

        listbox = tk.Listbox(self.manual_selection_window, exportselection=False)
        listbox.grid(row=row, column=2)
        setattr(self, f"{log_name.lower()}_listbox", listbox)
        self.update_listbox(log_name, search_var)

    def update_listbox(self, log_name, search_var):
        listbox = getattr(self, f"{log_name.lower()}_listbox")
        listbox.delete(0, tk.END)
        search_text = search_var.get().lower()
        for col in self.dataframe.columns:
            if search_text in col.lower():
                listbox.insert(tk.END, col)

    def confirm_log_selection(self):
        try:
            self.depth = self.dataframe['DEPT']
            self.dtc = self.dataframe[self.dtc_listbox.get(tk.ACTIVE)]
            self.dts = self.dataframe[self.dts_listbox.get(tk.ACTIVE)]
            self.rhob = self.dataframe[self.rhob_listbox.get(tk.ACTIVE)]
            self.progress_text.insert(tk.END, "Manual log selection successful.\n"
                                              "Required logs for dynamic geomechanics calculation loaded.\n")
        except KeyError as e:
            messagebox.showerror("Error", f"Invalid log selection: {e}")

    def calculate_moduli(self):
        # Convert DTC and DTS from microseconds/foot to m/s
        self.vp = (1 / self.dtc) * 1e6 * 0.3048  # V_P in m/s
        self.vs = (1 / self.dts) * 1e6 * 0.3048  # V_S in m/s
        # Calculate velocity ratio
        self.velocity_ratio = self.vp / self.vs
        # Convert RHOB from g/cc to kg/m³
        self.rho = self.rhob * 1000  # density in kg/m³
        # Calculate 1st Lamé parameter
        self.lame_parameter = self.rho * (self.vp ** 2 - 2 * self.vs ** 2)
        # Calculate Young's modulus
        self.youngs_modulus = (self.rho * self.vs ** 2) * (3 * self.vp ** 2 - 4 * self.vs ** 2) / (
                self.vp ** 2 - self.vs ** 2)
        # Calculate Shear modulus
        self.shear_modulus = self.rho * self.vs ** 2
        # Calculate Bulk modulus
        self.bulk_modulus = self.rho * (self.vp ** 2 - (4.0 / 3) * self.vs ** 2)
        # Calculate Poisson's ratio
        self.poisson_ratio = 0.5 * ((self.vp ** 2 - 2 * self.vs ** 2) / (self.vp ** 2 - self.vs ** 2))

        # Conversion factor from Pa to psi
        conversion_factor = 0.00014503774

        # Convert moduli to psi
        self.lame_parameter *= conversion_factor
        self.youngs_modulus *= conversion_factor
        self.shear_modulus *= conversion_factor
        self.bulk_modulus *= conversion_factor

        # Store results in dataframe
        self.dataframe['VRATIO'] = self.velocity_ratio
        self.dataframe['L_DYN'] = self.lame_parameter
        self.dataframe['E_DYN'] = self.youngs_modulus
        self.dataframe['G_DYN'] = self.shear_modulus
        self.dataframe['K_DYN'] = self.bulk_modulus
        self.dataframe['NU_DYN'] = self.poisson_ratio
        result_text = f"Dynamic elastic moduli calculated.\n"
        self.progress_text.insert(tk.END, result_text)

    def open_overburden_pressure(self):
        pob_window = tk.Toplevel(self.root)
        pob_window.title("Overburden Pressure Calculation")

        # Option 1: Input for Overburden Pressure Gradient
        tk.Label(pob_window, text="Option 1: Using a given gradient").grid(row=0, column=0, sticky="w")

        tk.Label(pob_window, text="Enter Overburden Pressure Gradient (psi/ft):").grid(row=1, column=0, sticky="w")
        gradient_entry = tk.Entry(pob_window)
        gradient_entry.grid(row=1, column=1)
        calc_pob_gradient_button = tk.Button(pob_window, text="Calculate POB",
                                             command=lambda: self.calculate_pob_from_gradient(gradient_entry.get()))
        calc_pob_gradient_button.grid(row=1, column=2)

        # Option 2: Input for Missed Density
        tk.Label(pob_window, text="Option 2: Using density log").grid(row=2, column=0, sticky="w")

        tk.Label(pob_window, text="Enter Density (g/cc) of missing interval:").grid(row=3, column=0, sticky="w")
        density_entry = tk.Entry(pob_window)
        density_entry.grid(row=3, column=1)
        calc_pob_density_button = tk.Button(pob_window, text="Calculate POB",
                                            command=lambda: self.calculate_pob_from_density(density_entry.get()))
        calc_pob_density_button.grid(row=3, column=2)

    def calculate_pob_from_gradient(self, gradient):
        try:
            gradient = float(gradient)
            self.pob = self.depth * gradient

            # Store results in dataframe
            self.dataframe['POB_DYN'] = self.pob
            result_text = f"Overburden Pressure calculated.\n"
            self.progress_text.insert(tk.END, result_text)
        except ValueError:
            messagebox.showerror("Error", "Invalid gradient input.")

    def calculate_pob_from_density(self, missed_density):
        try:
            self.rhob_miss = float(missed_density)
            self.calculate_overburden_pressure()
        except ValueError:
            messagebox.showerror("Error", "Invalid density input.")

    def calculate_overburden_pressure(self):
        # Find the index of the first non-null value in rhob
        first_valid_index = self.rhob.first_valid_index()
        # Calculate overburden pressure gradient in psi/ft
        if not self.depth[first_valid_index:].empty:
            # Use the corresponding depth value
            starting_depth = self.depth[first_valid_index]
            # Convert bulk density from g/cc to lb/ft³
            bulk_density_lbf_ft3 = self.rhob * 62.43
            bulk_density_miss_lbf_ft3 = self.rhob_miss * 62.43
            # Calculate overburden pressure for the missing section in lb/ft²
            obp_missing_lbf_ft2 = bulk_density_miss_lbf_ft3 * starting_depth
            # Calculate overburden pressure in lb/ft² for the rest of the depths
            obp_lbf_ft2 = np.cumsum(
                bulk_density_lbf_ft3[first_valid_index:] * np.diff(self.depth[first_valid_index:],
                                                                   prepend=starting_depth))
            # Total overburden pressure in lb/ft²
            obp_total_lbf_ft2 = obp_missing_lbf_ft2 + obp_lbf_ft2
            # Convert overburden pressure to psi
            self.pob = obp_total_lbf_ft2 * 0.00694444
            self.pob_gradient = self.pob / self.depth

            # Store results in dataframe
            self.dataframe['POB_DYN'] = self.pob
            self.dataframe['POBG_DYN'] = self.pob_gradient
            result_text = f"Overburden Pressure calculated.\n"
            self.progress_text.insert(tk.END, result_text)
        else:
            messagebox.showerror("Error", "Depth data is empty or invalid.")
            return

    def open_pore_pressure(self):
        if self.depth is None or self.dtc is None or self.rhob is None:
            messagebox.showerror("Error", "Depth, DTC, or RHOB data not loaded.")
            return
        if self.pob is None:
            messagebox.showerror("Error", "Overburden Pressure is not calculated.")
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Draw Normal Compaction Trendline and Calculate Pore Pressure")

        # Entry for hydrostatic gradient
        tk.Label(plot_window, text="Enter Hydrostatic Gradient (psi/ft):").pack()
        hydro_gradient_entry = tk.Entry(plot_window)
        hydro_gradient_entry.pack()

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()
        ax.semilogy(self.depth, self.dtc, label='DTC vs Depth')  # Semi-log plot for DTC
        line, = ax.plot([], [], 'r-', linewidth=2)  # Line object for the trendline
        points = []

        def onclick(event):
            if event.inaxes != ax: return
            if len(points) == 2:  # Reset points if two points were already selected
                points.clear()
                line.set_data([], [])
                fig.canvas.draw()

            points.append((event.xdata, np.log10(event.ydata)))  # Log-transform the y-coordinate
            if len(points) == 2:
                # Calculate the trendline in log space
                coef = np.polyfit([points[0][0], points[1][0]], [points[0][1], points[1][1]], 1)
                self.nct = coef  # Save coefficients for further calculations

                # Create a line in linear space
                log_line_y = np.polyval(coef, self.depth)
                line.set_data(self.depth, 10 ** log_line_y)  # Transform back to linear space
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)

        def calculate_pore_pressure():
            try:
                hydro_gradient = float(hydro_gradient_entry.get())
                hydrostatic_pressure = hydro_gradient * self.depth
                self.norm_dtc = 10 ** np.polyval(self.nct, self.depth)
                self.pp = self.pob - (self.pob - hydrostatic_pressure) * (self.norm_dtc / self.dtc) ** 3
                self.ppg = self.pp / self.depth
                # Display pore pressure calculation results
                self.dataframe['PP_DYN'] = self.pp
                self.dataframe['PPG_DYN'] = self.ppg
                result_text = f"Pore Pressure calculated.\n"
                self.progress_text.insert(tk.END, result_text)
            except ValueError:
                messagebox.showerror("Error", "Invalid hydrostatic gradient input.")

        # Button for calculating pore pressure
        calculate_button = tk.Button(plot_window, text="Calculate Pore Pressure", command=calculate_pore_pressure)
        calculate_button.pack()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        plot_window.mainloop()

    def open_biot(self):
        biot_window = tk.Toplevel(self.root)
        biot_window.title("Biot's Coefficient Calculation")

        # Dropdown for Total Porosity Log
        tk.Label(biot_window, text="Select Total Porosity Log:").grid(row=0, column=0, sticky="w")
        porosity_var = tk.StringVar()
        porosity_log_combobox = ttk.Combobox(biot_window, textvariable=porosity_var)
        porosity_log_combobox['values'] = self.dataframe.columns.tolist()
        porosity_log_combobox.grid(row=0, column=1)
        porosity_log_combobox.bind('<<ComboboxSelected>>', self.update_phit)

        # Option 1: Calculate Biot's Coefficient Using Empirical Relationships
        tk.Label(biot_window, text="Option 1: Calculate Biot's using Empirical Equation").grid(row=2, column=0,
                                                                                               sticky="w")
        calc_biot_empirical_button = tk.Button(biot_window, text="Calculate Biot's (Empirical)",
                                               command=self.calculate_biot_empirical)
        calc_biot_empirical_button.grid(row=3, column=0)

        # Option 2: Calculate Biot's Coefficient Using Sonic Logs
        tk.Label(biot_window, text="Option 2: Calculate Biot's using Density and Sonic Logs").grid(row=4, column=0,
                                                                                                   sticky="w")
        calc_biot_sonic_button = tk.Button(biot_window, text="Calculate Biot's (Sonic)",
                                           command=self.calculate_biot_sonic)
        calc_biot_sonic_button.grid(row=5, column=0)

    def update_phit(self, event):
        selected_log = event.widget.get()
        if selected_log in self.dataframe.columns:
            self.phit = self.dataframe[selected_log]
            self.progress_text.insert(tk.END, f"Total Porosity Log '{selected_log}' loaded successfully.\n")
        else:
            messagebox.showerror("Error", "Selected log not found in dataframe.")

    def calculate_biot_empirical(self):
        self.biot = 0.62 + 0.935 * self.phit
        self.dataframe['BIOT_DYN'] = self.biot
        result_text = f"Biot's coefficient calculated.\n"
        self.progress_text.insert(tk.END, result_text)

    def calculate_biot_sonic(self):
        # Calculate Matrix Density in kg/m³
        self.rhoma = (self.rho - self.phit) / (1 - self.phit)

        # Convert dtc and dts from microsec/ft to sec/m
        self.dtcma = (self.dtc - self.phit * self.dtcw) / (1 - self.phit)
        self.dtsma = self.dts / (1 - self.phit)

        # Calculate velocities in m/s
        vp_matrix = (1 / self.dtcma) * 1e6 * 0.3048  # Vp of matrix
        vs_matrix = (1 / self.dtsma) * 1e6 * 0.3048  # Vs of matrix

        # Calculate Matrix Bulk Modulus in Pascals (Pa)
        self.matrix_bulk_modulus = self.rhoma * (vp_matrix ** 2 - (4.0 / 3) * (vs_matrix ** 2))

        # Convert Matrix Bulk Modulus from Pa to psi
        pa_to_psi = 0.0001450377
        self.matrix_bulk_modulus *= pa_to_psi

        # Calculate Biot's coefficient
        self.biot = 1 - self.bulk_modulus / self.matrix_bulk_modulus

        # Store results in dataframe
        self.dataframe['BIOT_DYN'] = self.biot
        result_text = f"Biot's coefficient calculated.\n"
        self.progress_text.insert(tk.END, result_text)

    def calculate_min_hstress(self):
        if not self.pp.empty and not self.poisson_ratio.empty and not self.pob.empty and not self.biot.empty:
            self.min_hstress = ((self.poisson_ratio / (1 - self.poisson_ratio)) * (self.pob - self.biot * self.pp) +
                                self.biot * self.pp) + self.pext
            self.min_hstress_90percent = self.min_hstress * 0.9
            self.injectable = (self.min_hstress_90percent - self.pp)

            # Fracture Pressure for isotropic reservoir (Basic Model)
            self.pfrac = (self.pob - self.pp) * (self.poisson_ratio / (1 - self.poisson_ratio)) + self.pp
            self.frac_grad = self.pfrac / self.depth

            self.dataframe['MINHS_DYN'] = self.min_hstress
            self.dataframe['MINHS90_DYN'] = self.min_hstress_90percent
            self.dataframe['INJECTABLE_DYN'] = self.injectable
            self.dataframe['PFRAC_DYN'] = self.pfrac
            self.dataframe['FG_DYN'] = self.frac_grad

            result_text = f"Frac Pressure and Closure Stress calculated.\n"
            self.progress_text.insert(tk.END, result_text)

    def save_dynamic_results(self):
        if not self.file_path or self.dataframe is None:
            messagebox.showerror("Error", "No data to save.")
            return

        # Extracting filename without extension
        base_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        # Creating a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Constructing new filename
        new_filename = f"{base_filename}_geomechanics_{timestamp}"

        try:
            if self.file_path.lower().endswith('.csv'):
                self.dataframe.to_csv(os.path.join(self.output_dir, f'{new_filename}.csv'))
            elif self.file_path.lower().endswith('.xlsx'):
                self.dataframe.to_excel(os.path.join(self.output_dir, f'{new_filename}.xlsx'))
            elif self.file_path.lower().endswith('.las'):
                las = lasio.read(self.file_path)
                for col in self.dataframe.columns:
                    # Retrieve unit and description, or use defaults if not defined
                    unit = self.curve_attributes.get(col, {}).get('unit', '')
                    desc = self.curve_attributes.get(col, {}).get('desc', col)
                    las.append_curve(col, self.dataframe[col].values, unit=unit, descr=desc)
                las.write(os.path.join(self.output_dir, f'{new_filename}.las'))
            messagebox.showinfo("Success", "Data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")


if __name__ == "__main__":
    app = GeoMechanics
