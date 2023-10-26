import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk

import pandas as pd
from matplotlib import pyplot as plt

from modules.cmr_permeability import GaussianDecomposition, perform_gaussian_decomposition
from modules.electrofacies import electrofacies
from modules.graphs import LogViewer
from modules.las_io import LasIO
from modules.utils import ColorCoding as cc
import json


class LogCalculator(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.form_checkboxes = None
        self.clustering_params = None
        self.cluster_range = None
        self.n_components = None
        self.clustering_methods = None
        self.t2_browse_button = None
        self.grid(sticky=(tk.N, tk.W, tk.E, tk.S))  # Embed it into the provided frame
        self.create_widgets()

        self.lithology_color_coding = './data/color_code/lithology_color_code.xml'
        self.formations = []  # List to store selected formations
        self.template_xml_path = None
        self.log = []

        self.fluid_properties_params = ''
        self.multimineral_params = ''

    def create_widgets(self):
        # LAS File Entry
        self.las_file_path = tk.StringVar()
        self.las_entry = tk.Entry(self, textvariable=self.las_file_path, width=40)
        self.las_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky='w')
        tk.Label(self, text="LAS File Path:").grid(row=0, column=0, padx=10, pady=5)
        tk.Button(self, text="Browse", command=self.browse_las).grid(row=0, column=3)

        # Tops File Entry
        self.tops_file_path = tk.StringVar()
        self.tops_entry = tk.Entry(self, textvariable=self.tops_file_path, width=40)
        self.tops_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky='w')
        tk.Label(self, text="Tops File Path:").grid(row=1, column=0, padx=10, pady=5)
        tk.Button(self, text="Browse", command=self.browse_tops).grid(row=1, column=3)

        # Start Depth Entry
        self.start_depth = tk.DoubleVar()
        self.start_depth.set(0)
        self.start_depth_entry = tk.Entry(self, textvariable=self.start_depth, width=10)
        self.start_depth_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')
        tk.Label(self, text="Start Depth (MD, ft):").grid(row=2, column=0, padx=10, pady=5, )

        # End Depth Entry
        self.end_depth = tk.DoubleVar()
        self.end_depth.set(5000)
        self.end_depth_entry = tk.Entry(self, textvariable=self.end_depth, width=10)
        self.end_depth_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')
        tk.Label(self, text="End Depth (MD, ft):").grid(row=3, column=0, padx=10, pady=5)

        # Select Formations Button (added under depth inputs)

        self.select_formations_button = tk.Button(self, text="Select", command=self.select_formations,
                                                  state=tk.DISABLED)
        self.select_formations_button.grid(row=2, column=3, padx=10, pady=5)
        tk.Label(self, text="Select Formations:").grid(row=2, column=2, padx=10, pady=5)

        # Fluid Property Separator
        ttk.Separator(self, orient="horizontal").grid(row=4, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        # Calculate Fluid Properties Checkbox
        self.calculate_fluid_properties = tk.BooleanVar()
        self.calculate_fluid_properties_checkbox = tk.Checkbutton(self, text="Calculate Fluid Properties",
                                                                  variable=self.calculate_fluid_properties,
                                                                  command=self.toggle_parameter_list)
        self.calculate_fluid_properties_checkbox.grid(row=5, column=0, columnspan=1, padx=10, pady=5)

        # Fluid Properties Parameter Selection (Combobox)
        self.fluid_properties_params_options = self.load_fluid_properties_params()
        # self.fluid_properties_params = tk.StringVar()
        self.fluid_properties_params_dropdown = ttk.Combobox(self, values=self.fluid_properties_params_options)
        self.fluid_properties_params_dropdown.grid(row=5, column=3, padx=10, pady=10)
        self.fluid_properties_params_dropdown.set(self.fluid_properties_params_options[0])  # Set default value

        tk.Label(self, text="Select Fluid Properties Parameter:").grid(row=5, column=2, padx=10, pady=10)
        self.fluid_properties_params_dropdown.config(state=tk.DISABLED)  # Initially disabled

        # Multimineral Model Separator
        ttk.Separator(self, orient="horizontal").grid(row=6, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        # Multimineral Model Calculation Checkbox
        self.calculate_multimineral_model = tk.BooleanVar()
        self.calculate_multimineral_model.set(False)
        self.calculate_multimineral_model_checkbox = tk.Checkbutton(self, text="Calculate Multimineral Model",
                                                                    variable=self.calculate_multimineral_model,
                                                                    command=self.toggle_multimineral_params_list)
        self.calculate_multimineral_model_checkbox.grid(row=7, column=0, columnspan=1, padx=10, pady=5)

        # Multimineral Parameters Dropdown (initially disabled)
        self.multimineral_params_options = self.load_multimineral_params_options()
        # self.multimineral_params = tk.StringVar()
        self.multimineral_params_dropdown = ttk.Combobox(self, values=self.multimineral_params_options,
                                                         state=tk.DISABLED)
        self.multimineral_params_dropdown.set(self.multimineral_params_options[0])
        self.multimineral_params_dropdown.grid(row=7, column=3, padx=10, pady=5)
        tk.Label(self, text="Select Multimineral Parameters:").grid(row=7, column=2, padx=10, pady=5)

        # Electrofacies Separator
        ttk.Separator(self, orient="horizontal").grid(row=8, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        # Electrofacies Calculation Checkbox
        self.calculate_electrofacies = tk.BooleanVar()
        self.calculate_electrofacies_checkbox = tk.Checkbutton(self, text="Calculate Electrofacies",
                                                               variable=self.calculate_electrofacies,
                                                               command=self.toggle_electrofacies_config_button)
        self.calculate_electrofacies_checkbox.grid(row=9, column=0, columnspan=1, padx=10, pady=5)

        # Configuration Button for Electrofacies Parameters (initially disabled)
        self.configure_electrofacies_button = tk.Button(self, text="Configure Parameters",
                                                        command=self.open_electrofacies_config_window,
                                                        state=tk.DISABLED)
        self.configure_electrofacies_button.grid(row=9, column=1, padx=10, pady=5)

        # Output Separator
        ttk.Separator(self, orient="horizontal").grid(row=10, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        self.output_files = tk.BooleanVar(value=True)
        self.output_files_checkbox = tk.Checkbutton(self, text="Export output files",
                                                    variable=self.output_files,
                                                    command=self.toggle_output_files_checkbox)
        self.output_files_checkbox.grid(row=11, column=0, columnspan=1, padx=10, pady=5)

        # Output Format Listbox (initially on)
        self.output_files_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, height=3)
        self.output_files_listbox.grid(row=11, column=3, columnspan=1, padx=10, pady=5)

        self.output_files_options = ['.las', '.csv', '.xlsx']
        for idx, option in enumerate(self.output_files_options):
            self.output_files_listbox.insert(tk.END, option)
            if option == '.las':
                self.output_files_listbox.select_set(idx)  # Set default selection to .las
        self.output_files_listbox.config(state=tk.NORMAL)

        tk.Label(self, text="Select output format (default .las):").grid(row=11, column=2, padx=10, pady=5)

        # LogViewer Separator
        ttk.Separator(self, orient="horizontal").grid(row=13, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        self.log_display = tk.BooleanVar()
        self.log_display_checkbox = tk.Checkbutton(self, text="Display logs after calculation?",
                                                   variable=self.log_display,
                                                   command=self.toggle_xml_widgets)
        self.log_display_checkbox.grid(row=14, column=0, columnspan=1, padx=10, pady=5)

        # XML Template Selection
        self.template_options = ['raw', 'full', 'lithofacies', 'electrofacies', 'salinity', 'permeability']
        self.xml_template = tk.StringVar()
        self.xml_template.set(self.template_options[0])
        self.template_menu = tk.OptionMenu(self, self.xml_template, *self.template_options)
        self.template_menu.config(state=tk.DISABLED)
        self.template_menu.grid(row=15, column=1, padx=10, pady=10)
        tk.Label(self, text="Select XML Template:").grid(row=15, column=0, padx=10, pady=5)
        self.xml_template.trace_add('write', self.toggle_t2_entry)

        # XML Template loading if not using the default templates
        tk.Label(self, text="or Load Your XML:").grid(row=15, column=2, padx=10, pady=5)
        self.browse_button = tk.Button(self, text="Browse", command=self.browse_xmls, state=tk.DISABLED)
        self.browse_button.grid(row=15, column=3)

        # T2 File Path Entry (Permeability Option)
        self.t2_file_path = tk.StringVar()
        self.t2_entry = tk.Entry(self, textvariable=self.t2_file_path, state=tk.DISABLED)
        self.t2_entry.grid(row=16, column=1, padx=10, pady=10)
        tk.Label(self, text="CMR T2 File Path:").grid(row=16, column=0, padx=10, pady=5)
        self.t2_browse_button = tk.Button(self, text="Browse", command=self.browse_t2, state=tk.DISABLED)
        self.t2_browse_button.grid(row=16, column=2)

        # Display Decomposition Checkbox (Permeability Option)
        self.display_decomp = tk.BooleanVar()
        self.display_decomp.set(False)
        self.display_decomp_checkbox = tk.Checkbutton(self, text="Display Decomposition Result",
                                                      variable=self.display_decomp, state=tk.DISABLED)
        self.display_decomp_checkbox.grid(row=16, column=3, columnspan=2, padx=10, pady=5)

        # Run separator
        ttk.Separator(self, orient="horizontal").grid(row=17, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        # Run Button
        tk.Button(self, text="Run", command=self.run_processing).grid(row=18, column=1, columnspan=2, padx=10, pady=10)

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
            self.load_las_file(file_path)

    def browse_xmls(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml")])
        if file_path:
            self.template_xml_path = file_path

    def browse_tops(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.tops_file_path.set(file_path)
            # Load tops file immediately
            self.load_tops_file(file_path, self.las_file_path.get())

    def load_las_file(self, las_file_path):
        # Check if both LAS and tops files are loaded, then enable the "Select Formations" button
        if las_file_path and self.tops_file_path.get():
            self.select_formations_button.config(state=tk.NORMAL)

    def load_tops_file(self, tops_file_path, load_file_path):
        # Load tops using LasIO
        log = LasIO(load_file_path)
        log.load_tops(csv_path=tops_file_path, depth_type='MD', source=None)
        # Check if both LAS and tops files are loaded, then enable the "Select Formations" button
        if self.las_file_path.get() and tops_file_path:
            self.select_formations_button.config(state=tk.NORMAL)

    def select_formations(self):
        # Get formations from loaded tops
        log = LasIO(self.las_file_path.get())
        log.load_tops(csv_path=self.tops_file_path.get(), depth_type='MD', source=None)
        formations_to_select = list(log.tops.keys())

        # Create a new top-level window for the dialog
        formations_dialog = tk.Toplevel(self.master)
        formations_dialog.title("Select Formations")
        formations_dialog.geometry("400x300")

        def apply_and_close():
            # Update self.formations with the selected formations
            selected_indices = listbox.curselection()
            self.formations = [listbox.get(index) for index in selected_indices]
            print("Selected Formations:", self.formations)
            formations_dialog.destroy()

        def cancel_and_close():
            formations_dialog.destroy()

        scrollbar = ttk.Scrollbar(formations_dialog, orient="vertical")
        listbox = tk.Listbox(formations_dialog, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
        scrollbar.config(command=listbox.yview)

        for formation in formations_to_select:
            listbox.insert(tk.END, formation)
            if formation in self.formations:
                listbox.selection_set(formations_to_select.index(formation))

        listbox.grid(row=0, column=0, columnspan=2, sticky="nsew")
        scrollbar.grid(row=0, column=2, sticky="ns")

        # Create a frame for the buttons
        button_frame = tk.Frame(formations_dialog)

        ok_button = tk.Button(button_frame, text="OK", command=apply_and_close)
        ok_button.grid(row=0, column=0, padx=5, pady=10)

        cancel_button = tk.Button(button_frame, text="Cancel", command=cancel_and_close)
        cancel_button.grid(row=0, column=1, padx=5, pady=10)

        # Place the button frame at the bottom
        button_frame.grid(row=1, column=0, columnspan=2)

        # Configure row and column weights for resizing
        formations_dialog.grid_rowconfigure(0, weight=1)
        formations_dialog.grid_columnconfigure(0, weight=1)

        # Wait for the dialog to close
        self.wait_window(formations_dialog)

    def load_fluid_properties_params(self):
        # Load options for fluid properties parameters from the specified CSV file and column
        local_path = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(local_path, '../data/calculation_pars', 'fluid_properties.csv')

        try:
            df = pd.read_csv(csv_path)
            options = df['name'].tolist()
            return options
        except Exception as e:
            print(f"Error loading fluid properties parameters from CSV: {e}")
            return ['default', 'option1', 'option2', 'option3']  # Provide default options in case of an error

    def load_multimineral_params_options(self):
        # Define the path to the CSV file containing the multimineral parameters
        local_path = os.path.dirname(__file__)  # Replace with the path to your script
        csv_path = os.path.join(local_path, '../data/calculation_pars', 'multimineral_parameters.csv')

        try:
            df = pd.read_csv(csv_path)
            options = df['name'].tolist()
            return options
        except Exception as e:
            print(f"Error loading multimineral model parameters from CSV: {e}")
            return ['default', 'option1', 'option2', 'option3']  # Provide default options in case of an error

    def open_electrofacies_config_window(self):
        # Create a new top-level window for configuring electrofacies parameters
        electrofacies_config_window = tk.Toplevel(self.master)
        electrofacies_config_window.title("Electrofacies Configuration")
        electrofacies_config_window.geometry("400x800")

        # Electrofacies Configuration Parameters
        self.n_components = 0.85
        self.clustering_methods = ['kmeans']
        self.cluster_range = (5, 20)
        self.clustering_params = {
            'kmeans': {'n_clusters': 15, 'n_init': 3},
            'dbscan': {'eps': 0.8, 'min_samples': 8},
            'affinity': {'random_state': 20, 'affinity': 'euclidean'},
            'optics': {'min_samples': 20, 'max_eps': 0.5, 'xi': 0.05},
            'agglom': {'n_clusters': 12},
            'fuzzy': {'n_clusters': 9}
        }
        print("params", self.clustering_params)

        # Create input fields and labels for electrofacies parameters
        tk.Label(electrofacies_config_window, text="Variance Boundary for PCA:").grid(row=0, column=0,
                                                                                      padx=10, pady=5)
        n_components_entry = tk.Entry(electrofacies_config_window)
        n_components_entry.grid(row=0, column=1, padx=10, pady=5)
        n_components_entry.insert(0, str(self.n_components))  # Set default value

        # Add a Listbox widget for clustering methods
        tk.Label(electrofacies_config_window, text="Clustering Methods:").grid(row=1, column=0,
                                                                               padx=10, pady=5)
        clustering_methods_listbox = tk.Listbox(electrofacies_config_window, selectmode=tk.MULTIPLE, height=6)
        clustering_methods_listbox.grid(row=1, column=1, padx=10, pady=5)

        # Add a scrollbar for the Listbox
        scrollbar = ttk.Scrollbar(electrofacies_config_window, orient="vertical",
                                  command=clustering_methods_listbox.yview)
        scrollbar.grid(row=1, column=2, sticky="ns")
        clustering_methods_listbox.config(yscrollcommand=scrollbar.set)

        # Populate the Listbox with clustering methods
        available_clustering_methods = ['kmeans', 'dbscan', 'affinity', 'agglom', 'fuzzy']
        for method in available_clustering_methods:
            clustering_methods_listbox.insert(tk.END, method)

        # Set the default selected methods (if any)
        for method in self.clustering_methods:
            index = available_clustering_methods.index(method)
            clustering_methods_listbox.selection_set(index)

        # Create a function to get the selected clustering methods
        def get_selected_clustering_methods():
            selected_methods = [clustering_methods_listbox.get(index) for index in
                                clustering_methods_listbox.curselection()]
            return selected_methods

        tk.Label(electrofacies_config_window, text="Cluster Range (min, max):").grid(row=2, column=0, padx=10, pady=5)
        cluster_range_entry = tk.Entry(electrofacies_config_window)
        cluster_range_entry.grid(row=2, column=1, padx=10, pady=5)
        cluster_range_entry.insert(0, str(self.cluster_range))  # Set default value

        # KMeans LabelFrame:
        kmeans_frame = tk.LabelFrame(electrofacies_config_window, text="KMeans Parameters", padx=5, pady=5)
        kmeans_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        tk.Label(kmeans_frame, text="n_clusters:").grid(row=0, column=1, padx=10, pady=5)
        kmeans_n_clusters_entry = tk.Entry(kmeans_frame)
        kmeans_n_clusters_entry.grid(row=0, column=2, padx=10, pady=5)
        kmeans_n_clusters_entry.insert(0, str(self.clustering_params['kmeans']['n_clusters']))  # Preload default value
        tk.Label(kmeans_frame, text="n_init:").grid(row=1, column=1, padx=10, pady=5)
        kmeans_n_init_entry = tk.Entry(kmeans_frame)
        kmeans_n_init_entry.grid(row=1, column=2, padx=10, pady=5)
        kmeans_n_init_entry.insert(0, str(self.clustering_params['kmeans']['n_init']))  # Preload default value

        # DBSCAN LabelFrame:
        dbscan_frame = tk.LabelFrame(electrofacies_config_window, text="DBSCAN Parameters", padx=5, pady=5)
        dbscan_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        tk.Label(dbscan_frame, text="eps:").grid(row=0, column=1, padx=10, pady=5)
        dbscan_eps_entry = tk.Entry(dbscan_frame)
        dbscan_eps_entry.grid(row=0, column=2, padx=10, pady=5)
        dbscan_eps_entry.insert(0, str(self.clustering_params['dbscan']['eps']))  # Preload default value
        tk.Label(dbscan_frame, text="min_samples:").grid(row=1, column=1, padx=10, pady=5)
        dbscan_min_samples_entry = tk.Entry(dbscan_frame)
        dbscan_min_samples_entry.grid(row=1, column=2, padx=10, pady=5)
        dbscan_min_samples_entry.insert(0,
                                        str(self.clustering_params['dbscan']['min_samples']))  # Preload default value

        # Affinity LabelFrame:
        affinity_frame = tk.LabelFrame(electrofacies_config_window, text="Affinity Parameters", padx=5, pady=5)
        affinity_frame.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        tk.Label(affinity_frame, text="random_state:").grid(row=0, column=1, padx=10, pady=5)
        affinity_random_state_entry = tk.Entry(affinity_frame)
        affinity_random_state_entry.grid(row=0, column=2, padx=10, pady=5)
        affinity_random_state_entry.insert(0, str(
            self.clustering_params['affinity']['random_state']))  # Preload default value

        tk.Label(affinity_frame, text="affinity:").grid(row=1, column=1, padx=5, pady=5)
        affinity_affinity_entry = tk.Entry(affinity_frame)
        affinity_affinity_entry.grid(row=1, column=2, padx=10, pady=5)
        affinity_affinity_entry.insert(0, self.clustering_params['affinity']['affinity'])  # Preload default value

        # Agglom LabelFrame:
        agglom_frame = tk.LabelFrame(electrofacies_config_window, text="Agglom Parameters", padx=5, pady=5)
        agglom_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        tk.Label(agglom_frame, text="n_clusters:").grid(row=0, column=1, padx=10, pady=5)
        agglom_n_clusters_entry = tk.Entry(agglom_frame)
        agglom_n_clusters_entry.grid(row=0, column=2, padx=10, pady=5)
        agglom_n_clusters_entry.insert(0, str(self.clustering_params['agglom']['n_clusters']))  # Preload default value

        # Fuzzy LabelFrame:
        fuzzy_frame = tk.LabelFrame(electrofacies_config_window, text="Fuzzy Parameters", padx=5, pady=5)
        fuzzy_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        tk.Label(fuzzy_frame, text="n_clusters:").grid(row=0, column=1, padx=10, pady=5)
        fuzzy_n_clusters_entry = tk.Entry(fuzzy_frame)
        fuzzy_n_clusters_entry.grid(row=0, column=2, padx=10, pady=5)
        fuzzy_n_clusters_entry.insert(0, str(self.clustering_params['fuzzy']['n_clusters']))  # Preload default value

        # Create a function to save the entered parameters and close the window
        def save_parameters_and_close():
            try:
                # Update the n_components attribute
                self.n_components = float(n_components_entry.get())

                # Update the cluster_range attribute
                self.cluster_range = eval(cluster_range_entry.get())

                # Fetch values for kmeans parameters
                kmeans_n_clusters = int(kmeans_n_clusters_entry.get())
                kmeans_n_init = int(kmeans_n_init_entry.get())

                # Fetch values for dbscan parameters
                dbscan_eps = float(dbscan_eps_entry.get())

                # Fetch values for affinity parameters
                affinity_random_state = int(affinity_random_state_entry.get())
                affinity_affinity = affinity_affinity_entry.get()

                # Fetch values for agglom parameters
                agglom_n_clusters = int(agglom_n_clusters_entry.get())

                # Fetch values for fuzzy parameters
                fuzzy_n_clusters = int(fuzzy_n_clusters_entry.get())

                # Update the clustering_params dictionary with fetched values
                self.clustering_params = {
                    'kmeans': {'n_clusters': kmeans_n_clusters, 'n_init': kmeans_n_init},
                    'dbscan': {'eps': dbscan_eps},  # Keep existing or default to empty dict
                    'affinity': {'random_state': affinity_random_state, 'affinity': affinity_affinity},
                    'agglom': {'n_clusters': agglom_n_clusters},
                    'fuzzy': {'n_clusters': fuzzy_n_clusters},
                    'optics': self.clustering_params.get('optics', {}),  # Keep existing or default to empty dict
                }

                # Update the clustering_methods attribute with the selected methods
                self.clustering_methods = get_selected_clustering_methods()

                # Close the configuration window
                electrofacies_config_window.destroy()

            except ValueError:
                # Handle errors related to conversion
                tk.messagebox.showerror("Error", "Invalid input. Please check your entries and try again.")

        # Create a "Save" button to save the parameters
        save_button = tk.Button(electrofacies_config_window, text="Save", command=save_parameters_and_close)
        save_button.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

    def toggle_parameter_list(self):
        # Enable or disable the Fluid Properties Parameter dropdown based on the checkbox state
        if self.calculate_fluid_properties.get():
            self.fluid_properties_params_dropdown.config(state=tk.NORMAL)
        else:
            self.fluid_properties_params_dropdown.config(state=tk.DISABLED)

    def toggle_multimineral_params_list(self):
        if self.calculate_multimineral_model.get():
            self.multimineral_params_dropdown.config(state=tk.NORMAL)
        else:
            self.multimineral_params_dropdown.config(state=tk.DISABLED)

    def toggle_electrofacies_config_button(self):
        # Enable or disable the "Configure Parameters" button based on the "Calculate Electrofacies" checkbox state
        if self.calculate_electrofacies.get():
            self.configure_electrofacies_button.config(state=tk.NORMAL)
        else:
            self.configure_electrofacies_button.config(state=tk.DISABLED)

    def toggle_output_files_checkbox(self):
        # Check the state of the checkbox
        if self.output_files.get():
            self.output_files_listbox.config(state=tk.NORMAL)  # Enable the Listbox
            if not self.output_files_listbox.curselection():  # If no items are currently selected
                default_index = self.output_files_options.index('.las')
                self.output_files_listbox.select_set(default_index)  # Default to .las
        else:
            self.output_files_listbox.config(state=tk.DISABLED)  # Disable the Listbox

    def toggle_xml_widgets(self):
        if self.log_display.get():
            # If log_display is checked
            self.template_menu.config(state=tk.NORMAL)
            self.xml_template.set(self.template_options[0])  # Optionally reset to default
        else:
            # If log_display is unchecked
            self.template_menu.config(state=tk.DISABLED)
            self.xml_template.set('')  # Optionally clear the current selection

        # Similarly, for the Browse button
        browse_button_state = tk.NORMAL if self.log_display.get() else tk.DISABLED
        self.browse_button.config(state=browse_button_state)

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
        xml_template = self.xml_template.get() if self.xml_template.get() else None
        t2_file_path = self.t2_file_path.get() if xml_template == 'permeability' else None
        display_decomp = self.display_decomp.get()
        self.multimineral_params = self.multimineral_params_dropdown.get()
        self.fluid_properties_params = self.fluid_properties_params_dropdown.get()

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

        # Load formation tops
        if tops_file_path:
            log.load_tops(csv_path=tops_file_path, depth_type='MD', source=None)
        else:
            messagebox.showinfo("Warning", "No formation tops provided. Calculation might be off!")
            log.load_tops(csv_path=None, depth_type='MD', source=None)

        # df = log.log_qc(start_depth, end_depth)

        # Auto aliasing log names
        log.aliasing()

        # Calculate formation fluid property parameters
        if self.calculate_fluid_properties.get():
            log.load_fluid_properties()
            log.formation_fluid_properties(formations=self.formations, parameter=self.fluid_properties_params)

        # Calculate multimineral model
        if self.calculate_multimineral_model.get():
            log.load_multimineral_parameters()
            log.formation_multimineral_model(formations=self.formations, parameter=self.multimineral_params)

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

        # Electrofacies Calculation
        if self.calculate_electrofacies.get():
            # Use these parameters for electrofacies calculation
            n_components = self.n_components
            clustering_methods = self.clustering_methods
            print("params", self.clustering_params)
            cluster_range = self.cluster_range
            clustering_params = self.clustering_params
            logs = [log]  # List of Log objects
            curves = []
            log_scale = ['RESSHAL_N', 'RESMED_N',
                         'RESDEEP_N']  # List of curve names to preprocess on a log scale (optional)
            curve_names = []  # List of names for output electrofacies curves (optional)
            output_template, _ = electrofacies(logs, self.formations, curves, log_scale=log_scale,
                                               n_components=n_components, curve_names=curve_names,
                                               clustering_methods=clustering_methods,
                                               clustering_params=clustering_params,
                                               template=xml_template,
                                               template_xml_path=self.template_xml_path,
                                               lithology_color_coding=self.lithology_color_coding,
                                               masking=masking)

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

        # Export converted data (raw) to either .csv or .xlsx
        las_file_name = os.path.splitext(os.path.basename(las_file_path))[0]
        current_time = datetime.now().strftime('%m%d%H%M')
        excel_output = f'./output/Petrophysics/{las_file_name}_Petrophysics_{current_time}.xlsx'
        csv_output = f'./output/Petrophysics/{las_file_name}_Petrophysics_{current_time}.csv'
        las_output = f'./output/Petrophysics/{las_file_name}_Petrophysics_{current_time}.las'

        if self.output_files.get():  # If the main checkbox for exporting is checked
            selected_indices = self.output_files_listbox.curselection()  # Returns a tuple of selected item indices

            for index in selected_indices:
                selected_format = self.output_files_listbox.get(index)

                if selected_format == '.xlsx':
                    log.export_excel(excel_output)
                elif selected_format == '.csv':
                    log.export_csv(csv_output)
                elif selected_format == '.las':
                    log.write(las_output)

        # Provide a message box indicating that the processing is completed
        messagebox.showinfo("Processing Completed", "Data processing completed!")

        if self.log_display.get():
            # View and modify logs
            viewer = LogViewer(log, template_defaults=xml_template, template_xml_path=self.template_xml_path, top=4500,
                               height=500, lithology_color_coding=color,
                               masking=masking)
            viewer.fig.set_size_inches(17, 11)

            # add well_name to title of LogViewer #

            viewer.fig.suptitle(well_name, fontweight='bold', fontsize=30)

            # add logo to top left corner #

            logo_im = plt.imread('./data/logo/ca_logo.png')
            logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
            logo_ax.imshow(logo_im)
            logo_ax.axis('off')

            viewer.show()


if __name__ == "__main__":
    app = LogCalculator()
    app.mainloop()
