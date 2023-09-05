from EDA.MoPanda.MoPanda.modules.las_io import LasIO
from EDA.MoPanda.MoPanda.modules.graphs import LogViewer
from EDA.MoPanda.MoPanda.modules.electrofacies import electrofacies
from EDA.MoPanda.MoPanda.modules.utils import ColorCoding as cc
import os

import pandas as pd
from EDA.MoPanda.MoPanda.modules.cmr_permeability import GaussianDecomposition, perform_gaussian_decomposition
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


# Import other required modules and classes

class LogCalculator(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.grid(sticky=(tk.N, tk.W, tk.E, tk.S))  # Embed it into the provided frame
        self.create_widgets()

        self.lithology_color_coding = './data/color_code/lithology_color_code.xml'

        self.t2_browse_button = tk.Button(self, text="Browse", command=self.browse_t2, state=tk.DISABLED)
        self.t2_browse_button.grid(row=5, column=2)  # Add this line

        # Display Decomposition Checkbox (Permeability Option)
        self.display_decomp = tk.BooleanVar()
        self.display_decomp.set(False)
        self.display_decomp_checkbox = tk.Checkbutton(self, text="Display Decomposition Result",
                                                      variable=self.display_decomp, state=tk.DISABLED)
        self.display_decomp_checkbox.grid(row=6, column=1, padx=10, pady=10)

        # Run Button
        tk.Button(self, text="Run", command=self.run_processing).grid(row=7, column=1, padx=10, pady=20)

    def create_widgets(self):
        # LAS File Entry
        self.las_file_path = tk.StringVar()
        self.las_entry = tk.Entry(self, textvariable=self.las_file_path)
        self.las_entry.grid(row=0, column=1, padx=10, pady=10)
        tk.Label(self, text="LAS File Path:").grid(row=0, column=0, padx=10, pady=10)
        tk.Button(self, text="Browse", command=self.browse_las).grid(row=0, column=2)

        # Tops File Entry
        self.tops_file_path = tk.StringVar()
        self.tops_entry = tk.Entry(self, textvariable=self.tops_file_path)
        self.tops_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Label(self, text="Tops File Path:").grid(row=1, column=0, padx=10, pady=10)
        tk.Button(self, text="Browse", command=self.browse_tops).grid(row=1, column=2)

        # Start Depth Entry
        self.start_depth = tk.DoubleVar()
        self.start_depth.set(3800)
        self.start_depth_entry = tk.Entry(self, textvariable=self.start_depth)
        self.start_depth_entry.grid(row=2, column=1, padx=10, pady=10)
        tk.Label(self, text="Start Depth:").grid(row=2, column=0, padx=10, pady=10)

        # End Depth Entry
        self.end_depth = tk.DoubleVar()
        self.end_depth.set(5000)
        self.end_depth_entry = tk.Entry(self, textvariable=self.end_depth)
        self.end_depth_entry.grid(row=3, column=1, padx=10, pady=10)
        tk.Label(self, text="End Depth:").grid(row=3, column=0, padx=10, pady=10)

        # XML Template Selection
        self.template_options = ['raw', 'full', 'lithofacies', 'electrofacies', 'salinity', 'permeability']
        self.xml_template = tk.StringVar()
        self.xml_template.set(self.template_options[0])
        self.template_menu = tk.OptionMenu(self, self.xml_template, *self.template_options)
        self.template_menu.grid(row=4, column=1, padx=10, pady=10)
        tk.Label(self, text="XML Template:").grid(row=4, column=0, padx=10, pady=10)
        self.xml_template.trace_add('write', self.toggle_t2_entry)

        # T2 File Path Entry (Permeability Option)
        self.t2_file_path = tk.StringVar()
        self.t2_entry = tk.Entry(self, textvariable=self.t2_file_path, state=tk.DISABLED)
        self.t2_entry.grid(row=5, column=1, padx=10, pady=10)
        tk.Label(self, text="T2 Distribution File Path:").grid(row=5, column=0, padx=10, pady=10)
        tk.Button(self, text="Browse", command=self.browse_t2, state=tk.DISABLED).grid(row=5, column=2)

        # Display Decomposition Checkbox (Permeability Option)
        self.display_decomp = tk.BooleanVar()
        self.display_decomp.set(False)
        self.display_decomp_checkbox = tk.Checkbutton(self, text="Display Decomposition Result",
                                                      variable=self.display_decomp, state=tk.DISABLED)
        self.display_decomp_checkbox.grid(row=6, column=1, padx=10, pady=10)

        # Run Button
        tk.Button(self, text="Run", command=self.run_processing).grid(row=7, column=1, padx=10, pady=20)

    def toggle_t2_entry(self, *args):
        # Enable or disable the T2 File Path entry based on the selected XML template
        if self.xml_template.get() == 'permeability':
            self.t2_entry.config(state=tk.NORMAL)
            self.display_decomp_checkbox.config(state=tk.NORMAL)
            self.t2_browse_button.config(state=tk.NORMAL)  # Added this line
        else:
            self.t2_entry.config(state=tk.DISABLED)
            self.display_decomp_checkbox.config(state=tk.DISABLED)
            self.t2_browse_button.config(state=tk.DISABLED)  # Added this line

    def browse_las(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if file_path:
            self.las_file_path.set(file_path)

    def browse_tops(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.tops_file_path.set(file_path)

    def browse_t2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            self.t2_file_path.set(file_path)

    def run_processing(self):
        # Get user inputs
        las_file_path = self.las_file_path.get()
        tops_file_path = self.tops_file_path.get()
        start_depth = self.start_depth.get()
        end_depth = self.end_depth.get()
        xml_template = self.xml_template.get()
        t2_file_path = self.t2_file_path.get() if xml_template == 'permeability' else None
        display_decomp = self.display_decomp.get()

        masking = {
            'status': False,
            'mode': 'white',
            'facies_to_drop': ['Silty Shale', 'Shaly Sandstone', 'Shale', 'Black Shale', 'Halite', 'Anhydrite',
                               'Gypsum',
                               'Anomaly', 'Else'],
            'curves_to_mask': ['SALINITY_N', 'RWA_N'],
        }

        # Load LAS file
        log = LasIO(las_file_path)

        # Load the color-label-lithology relationship
        color = cc().litho_color(self.lithology_color_coding)

        df = log.log_qc(start_depth, end_depth)

        # Auto aliasing log names
        log.aliasing()

        # Load formation tops
        log.load_tops(csv_path=tops_file_path, depth_type='MD', source='CA')

        # calculate permeability if permeability template is selected
        if xml_template == 'permeability':
            file_name = os.path.basename(t2_file_path)
            df = pd.read_excel(t2_file_path)
            num_components = 7

            log = perform_gaussian_decomposition(log, df, file_name, num_components=num_components)
            if display_decomp:
                gd = GaussianDecomposition(df)
                index = simpledialog.askinteger("Decomposition Index",
                                                "Enter the index number for decomposition result (0 to {}): ".format(
                                                    len(df) - 1), minvalue=0, maxvalue=len(df) - 1)

                gd.decomposition_single(index, num_components=num_components, auto_search=False)

        # predictor = WellLogPredictor(log)

        # Electrofacies
        logs = [log]  # List of Log objects
        formations = ['SKULL_CREEK_SH', 'LAKOTA_UPPER', 'LAKOTA_LOWER', 'MORRISON',
                      'DAYCREEK', 'FLOWERPOT_SH', 'LYONS', 'SUMNER_SATANKA',
                      'STONE_CORRAL']  # List of formation names (optional)
        # curves = []
        curves = ['CAL_N', 'RHOB_N', 'CGR_N', 'SP_N', 'NPHI_N', 'DPHI_N', 'PE_N', 'SGR_N',
                  'RESSHAL_N', 'RESDEEP_N', 'DTS_N', 'DTC_N', 'TCMR', 'T2LM', 'RHOMAA_N', 'UMAA_N',
                  'RWA_N']  # List of curve names (optional)
        log_scale = ['RESSHAL_N', 'RESDEEP_N']  # List of curve names to preprocess on a log scale (optional)
        n_components = 0.85  # Number of principal components to keep (optional)
        curve_names = []  # List of names for output electrofacies curves (optional)
        clustering_methods = ['kmeans', 'dbscan', 'affinity', 'agglom',
                              'fuzzy']  # List of clustering methods to be used (optional)
        cluster_range = (2, 10)
        clustering_params = {
            'kmeans': {'n_clusters': 12, 'n_init': 3},  # "n_clusters" is optional if auto optimization is wanted
            'dbscan': {'eps': 0.8, 'min_samples': 8},
            'affinity': {'random_state': 20, 'affinity': 'euclidean'},
            'optics': {'min_samples': 20, 'max_eps': 0.5, 'xi': 0.05},
            'agglom': {'n_clusters': 12},
            'fuzzy': {'n_clusters': 9}  # "n_clusters" is optional if auto optimization is wanted
        }
        if xml_template == 'electrofacies':
            output_template, _ = electrofacies(logs, formations, curves, log_scale=log_scale,
                                               n_components=n_components, curve_names=curve_names,
                                               clustering_methods=clustering_methods,
                                               clustering_params=clustering_params,
                                               template=xml_template,
                                               lithology_color_coding=self.lithology_color_coding,
                                               masking=masking)
        else:
            electrofacies(logs, formations, curves, log_scale=log_scale,
                          n_components=n_components, curve_names=curve_names,
                          clustering_methods=['kmeans'],
                          clustering_params=clustering_params,
                          template=xml_template,
                          lithology_color_coding=self.lithology_color_coding,
                          masking=masking)
        print(log.curves)

        # find way to name well, looking for well name#
        # or UWI or API #

        if len(log.well['WELL'].value) > 0:
            well_name = log.well['WELL'].value
        elif len(str(log.well['UWI'].value)) > 0:
            well_name = str(log.well['UWI'].value)
        elif len(log.well['API'].value) > 0:
            well_name = str(log.well['API'].value)
        else:
            well_name = 'UNKNOWN'
        well_name = well_name.replace('.', '')

        # View and modify logs
        viewer = LogViewer(log, template_defaults=xml_template, top=4500, height=500, lithology_color_coding=color,
                           masking=masking)
        viewer.fig.set_size_inches(17, 11)

        # add well_name to title of LogViewer #

        viewer.fig.suptitle(well_name, fontweight='bold', fontsize=30)

        # add logo to top left corner #

        logo_im = plt.imread('./logo/ca_logo.png')
        logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
        logo_ax.imshow(logo_im)
        logo_ax.axis('off')

        viewer.show()

        # Export converted data (raw) to either .csv or .xlsx
        las_file_name = os.path.splitext(os.path.basename(las_file_path))[0]
        excel_output = f'./output/{las_file_name}_{xml_template}.xlsx'
        las_output = f'./output/{las_file_name}_{xml_template}.las'

        log.export_excel(excel_output)
        log.write(las_output)

        # Provide a message box indicating that the processing is completed
        messagebox.showinfo("Processing Completed", "Data processing completed!")


if __name__ == "__main__":
    app = LogCalculator()
    app.mainloop()
