import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib import pyplot as plt

from modules.graphs import LogViewer
from modules.las_io import LasIO



class MaskingForScoping:
    def __init__(self, master, input_folder_var, masking_var, gr_filter_var):
        self.master = master
        self.input_folder_var = input_folder_var
        self.masking_var = masking_var
        self.gr_filter_var = gr_filter_var

        # Setting the window title and geometry
        self.master.title("Scoping")
        self.master.geometry("400x600")  # Adjust size as needed

        # Default values
        self.start_depth = 3000
        self.end_depth = 20000
        self.salinity_limit = 10000
        self.phi_limit = 0.15
        self.gr_limit = 80
        self.masking = {
            'status': False,
            'mode': 'white',
            'facies_to_drop': [],
            'curves_to_mask': [],
        }

        # Initialize the GUI components
        self.init_gui()

    def init_gui(self):
        # Labels and Entry fields

        # Input Folder Entry
        tk.Label(self.master, text="Input Folder:").grid(row=0, column=0, sticky="e")
        self.input_folder_entry = tk.Entry(self.master, textvariable=self.input_folder_var, state='readonly')
        self.input_folder_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky='we')
        # Button for selecting input folder
        self.folder_select_button = tk.Button(self.master, text="Select Folder", command=self.select_input_folder)
        self.folder_select_button.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(self.master, text="Start Depth:").grid(row=1, column=0, sticky="e")
        self.start_depth_entry = tk.Entry(self.master)
        self.start_depth_entry.grid(row=1, column=1)
        self.start_depth_entry.insert(0, str(self.start_depth))

        tk.Label(self.master, text="End Depth:").grid(row=2, column=0, sticky="e")
        self.end_depth_entry = tk.Entry(self.master)
        self.end_depth_entry.grid(row=2, column=1)
        self.end_depth_entry.insert(0, str(self.end_depth))

        tk.Label(self.master, text="Salinity Limit:").grid(row=3, column=0, sticky="e")
        self.salinity_limit_entry = tk.Entry(self.master)
        self.salinity_limit_entry.grid(row=3, column=1)
        self.salinity_limit_entry.insert(0, str(self.salinity_limit))

        tk.Label(self.master, text="Porosity Limit:").grid(row=4, column=0, sticky="e")
        self.phi_limit_entry = tk.Entry(self.master)
        self.phi_limit_entry.grid(row=4, column=1)
        self.phi_limit_entry.insert(0, str(self.phi_limit))

        tk.Label(self.master, text="Gamma Ray Limit:").grid(row=5, column=0, sticky="e")
        self.gr_limit_entry = tk.Entry(self.master)
        self.gr_limit_entry.grid(row=5, column=1)
        self.gr_limit_entry.insert(0, str(self.gr_limit))

        # Checkbuttons
        tk.Label(self.master, text="Masking:").grid(row=6, column=0, sticky="e")
        self.masking_check = tk.Checkbutton(self.master, variable=self.masking_var)
        self.masking_check.grid(row=6, column=1)

        tk.Label(self.master, text="Gamma Ray Filter:").grid(row=7, column=0, sticky="e")
        self.gr_filter_check = tk.Checkbutton(self.master, variable=self.gr_filter_var)
        self.gr_filter_check.grid(row=7, column=1)

        # Buttons
        self.process_button = tk.Button(self.master, text="Process LAS Files", command=self.process_las_files)
        self.process_button.grid(row=8, column=1, columnspan=2, pady=10)

        self.simple_viewer_button = tk.Button(self.master, text="Simple Scoping Log Viewer", command=self.simple_scoping_viewer)
        self.simple_viewer_button.grid(row=9, column=1, columnspan=2, pady=10)

        self.scoping_viewer_button = tk.Button(self.master, text="Scoping Log Viewer", command=self.scoping_viewer)
        self.scoping_viewer_button.grid(row=10, column=1, columnspan=2, pady=10)

    def select_input_folder(self):
        # Open a dialog to select a folder, and update the input_folder_var
        self.folder_selected = filedialog.askdirectory()
        if self.folder_selected:  # Check if a folder was selected
            self.input_folder_var.set(self.folder_selected)

    def scoping_viewer(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if file_path:
            log = LasIO(file_path)
            print(log.curves)
            # Display the log using the Scoping template defaults
            viewer = LogViewer(log, template_defaults='scoping', top=self.start_depth, height=1000, masking=self.masking)
            viewer.show()

    def simple_scoping_viewer(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
        if file_path:
            log = LasIO(file_path)
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

            # Display the log using the Scoping template defaults
            viewer = LogViewer(log, template_defaults='scoping_simple', top=self.start_depth, height=1000, masking=self.masking)
            viewer.fig.set_size_inches(8, 11)

            viewer.fig.suptitle(well_name, fontweight='bold', fontsize=14)

            # add logo to top left corner #

            logo_im = plt.imread('./data/logo/ca_logo.png')
            logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
            logo_ax.imshow(logo_im)
            logo_ax.axis('off')

            viewer.show()

    def scoping(self, log, start_depth, end_depth, gr_filter):

        # Target logs
        target_logs = ['SALINITY_N', 'POR_N']

        # qc the logs within assigned depth interval
        # log.log_qc(start_depth, end_depth)

        # Auto aliasing log names
        log.aliasing()

        # Auto turning off GR filter
        if 'SGR_N' not in log.curves:
            gr_filter = False

        # Calculate formation fluid property parameters
        log.load_fluid_properties()
        log.formation_fluid_properties(top=start_depth, bottom=end_depth, parameter='default')

        # Auto aliasing log names again for future use
        log.aliasing()

        df = log.df()
        df['DEPTH_INDEX'] = np.arange(0, len(log[0]))

        data = np.empty(len(log[0]))
        data[:] = np.nan
        if gr_filter:
            depth_index = df.loc[
                (df['SALINITY_N'] > self.salinity_limit) & (df['POR_N'] > self.phi_limit) & (
                        df['SGR_N'] < self.gr_limit), 'DEPTH_INDEX']
            target_logs.append('SGR_N')
            for curve in target_logs:
                data[depth_index] = df.loc[(df['SALINITY_N'] > self.salinity_limit) & (df['POR_N'] > self.phi_limit)& (
                        df['SGR_N'] < self.gr_limit), curve]
                log.append_curve(
                    f'{curve}_MASKED',
                    np.copy(data),
                    descr=f'Masked {curve}',
                )
        else:
            depth_index = df.loc[(df['SALINITY_N'] > self.salinity_limit) & (df['POR_N'] > self.phi_limit), 'DEPTH_INDEX']
            for curve in target_logs:
                data[depth_index] = df.loc[(df['SALINITY_N'] > self.salinity_limit) & (df['POR_N'] > self.phi_limit), curve]
                log.append_curve(
                    f'{curve}_MASKED',
                    np.copy(data),
                    descr=f'Masked {curve}',
                )

        return log

    def process_las_files(self):
        # Update parameters based on user input from the GUI
        self.start_depth = int(self.start_depth_entry.get())
        self.end_depth = int(self.end_depth_entry.get())
        self.salinity_limit = int(self.salinity_limit_entry.get())
        self.phi_limit = float(self.phi_limit_entry.get())
        self.gr_limit = int(self.gr_limit_entry.get())
        self.masking['status'] = self.masking_var.get()
        self.gr_filter = self.gr_filter_var.get()

        # Get user-selected input folder
        input_folder = self.folder_selected
        if not input_folder:
            messagebox.showwarning("Warning", "Please select an input folder.")
            return

        # Prepare the output folder
        output_folder = os.path.join(os.path.dirname(input_folder), 'Scoped wells', os.path.basename(input_folder))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process each LAS file in the input folder
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".las"):
                    try:
                        las_file_path = os.path.join(root, file)
                        log = LasIO(las_file_path)

                        # Apply masking and filtering if enabled
                        if self.masking.get('status'):
                            log = self.scoping(log, self.start_depth, self.end_depth, self.gr_filter)

                        # Check for necessary curves
                        if 'SALINITY_N' in log.curves and 'POR_N' in log.curves:
                            relative_path = os.path.relpath(las_file_path, input_folder)
                            destination_folder = os.path.join(output_folder, os.path.dirname(relative_path))
                            if not os.path.exists(destination_folder):
                                os.makedirs(destination_folder)
                            destination_file = os.path.join(destination_folder, file)
                            log.write(destination_file, mnemonics_header=True, data_section_header="~A")
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")

        messagebox.showinfo("Process Complete", "LAS files processing is complete!")


