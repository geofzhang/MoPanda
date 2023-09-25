import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib import pyplot as plt

from EDA.MoPanda.MoPanda.modules.graphs import LogViewer
from EDA.MoPanda.MoPanda.modules.las_io import LasIO

# Default values for the variables
start_depth = 3000
end_depth = 4000
salinity_limit = 10000
phi_limit = 0.15
gr_limit = 80
masking = {
    'status': False,
    'mode': 'white',
    'facies_to_drop': [],
    'curves_to_mask': [],
}


class MissingLogException(Exception):
    pass


def scoping_viewer():
    file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
    if file_path:
        log = LasIO(file_path)
        print(log.curves)
        # Display the log using the Scoping template defaults
        viewer = LogViewer(log, template_defaults='scoping', top=start_depth, height=1000, masking=masking)
        viewer.show()


def simple_scoping_viewer():
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
        viewer = LogViewer(log, template_defaults='scoping_simple', top=start_depth, height=1000, masking=masking)
        viewer.fig.set_size_inches(8, 11)

        viewer.fig.suptitle(well_name, fontweight='bold', fontsize=14)

        # add logo to top left corner #

        logo_im = plt.imread('./logo/ca_logo.png')
        logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
        logo_ax.imshow(logo_im)
        logo_ax.axis('off')

        viewer.show()


def scoping(log, start_depth, end_depth, gr_filter):
    # Auto turning off GR filter
    if 'SGR_N' not in log.curves:
        gr_filter = False

    # Target logs
    target_logs = ['SALINITY_N', 'POR_N']
    # qc the logs within assigned depth interval
    # log.log_qc(start_depth, end_depth)

    # Auto aliasing log names
    log.aliasing()

    # Calculate formation fluid property parameters
    log.load_fluid_properties()
    log.formation_fluid_properties(top=start_depth, bottom=end_depth, parameter='default')

    # print(log.curves)

    df = log.df()
    df['DEPTH_INDEX'] = np.arange(0, len(log[0]))

    data = np.empty(len(log[0]))
    data[:] = np.nan
    if gr_filter:
        depth_index = df.loc[
            (df['SALINITY_N'] > salinity_limit) & (df['POR_N'] > phi_limit) & (
                    df['SGR_N'] < gr_limit), 'DEPTH_INDEX']
        for curve in target_logs.append('SGR_N'):
            data[depth_index] = df.loc[(df['SALINITY_N'] > salinity_limit) & (df['POR_N'] > phi_limit), curve]
            log.append_curve(
                f'{curve}_MASKED',
                np.copy(data),
                descr=f'Masked {curve}',
            )
    else:
        depth_index = df.loc[(df['SALINITY_N'] > salinity_limit) & (df['POR_N'] > phi_limit), 'DEPTH_INDEX']
        print(depth_index)
        for curve in target_logs:
            data[depth_index] = df.loc[(df['SALINITY_N'] > salinity_limit) & (df['POR_N'] > phi_limit), curve]
            log.append_curve(
                f'{curve}MASKED',
                np.copy(data),
                descr=f'Masked {curve}',
            )

    return log


def process_las_files():
    global masking, gr_filter  # Declare the variables as global within the function
    masking = tk.BooleanVar(value=True)
    gr_filter = tk.BooleanVar(value=False)

    # Get user-selected input folder
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    if not input_folder:
        messagebox.showwarning("Warning", "Please select an input folder.")
        return

    output_folder = os.path.join(os.path.dirname(input_folder), 'Scoped wells', os.path.basename(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get user-selected values from the GUI
    global start_depth, end_depth, salinity_limit, phi_limit, gr_limit
    start_depth = int(start_depth_entry.get())
    end_depth = int(end_depth_entry.get())
    salinity_limit = int(salinity_limit_entry.get())
    phi_limit = float(phi_limit_entry.get())
    gr_limit = int(gr_limit_entry.get())
    masking = masking_var.get()
    gr_filter = gr_filter_var.get()

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".las"):
                las_file_path = os.path.join(root, file)
                log = LasIO(las_file_path)
                if masking:
                    log = scoping(log, start_depth, end_depth, gr_filter)

                if 'SALINITY_N' in log.curves and 'POR_N' in log.curves:
                    relative_path = os.path.relpath(las_file_path, input_folder)
                    destination_folder = os.path.join(output_folder, os.path.dirname(relative_path))
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    destination_file = os.path.join(destination_folder, file)
                    log.write(destination_file, mnemonics_header=True, data_section_header="~A")

    messagebox.showinfo("Process Complete", "LAS files processing is complete!")


# Create the GUI
root = tk.Tk()
root.title("LAS Data Processing")
root.geometry("400x400")  # Set the window size to 400x400

# Labels
tk.Label(root, text="Input Folder:").grid(row=0, column=0, sticky="e")

# Entry fields
input_folder_var = tk.StringVar(value=os.getcwd())  # StringVar to store the selected input folder path
input_folder_entry = tk.Entry(root, textvariable=input_folder_var, state='readonly')
input_folder_entry.grid(row=0, column=1, columnspan=50, padx=5, pady=5, sticky='we')

tk.Label(root, text="Start Depth:").grid(row=1, column=0, sticky="e")
tk.Label(root, text="End Depth:").grid(row=2, column=0, sticky="e")
tk.Label(root, text="Salinity Limit:").grid(row=3, column=0, sticky="e")
tk.Label(root, text="Porosity Limit:").grid(row=4, column=0, sticky="e")
tk.Label(root, text="Gamma Ray Limit:").grid(row=5, column=0, sticky="e")
tk.Label(root, text="Masking:").grid(row=6, column=0, sticky="e")
tk.Label(root, text="Gamma Ray Filter:").grid(row=7, column=0, sticky="e")

# Entry fields
start_depth_entry = tk.Entry(root)
start_depth_entry.grid(row=1, column=1)
start_depth_entry.insert(0, str(start_depth))

end_depth_entry = tk.Entry(root)
end_depth_entry.grid(row=2, column=1)
end_depth_entry.insert(0, str(end_depth))

salinity_limit_entry = tk.Entry(root)
salinity_limit_entry.grid(row=3, column=1)
salinity_limit_entry.insert(0, str(salinity_limit))

phi_limit_entry = tk.Entry(root)
phi_limit_entry.grid(row=4, column=1)
phi_limit_entry.insert(0, str(phi_limit))

gr_limit_entry = tk.Entry(root)
gr_limit_entry.grid(row=5, column=1)
gr_limit_entry.insert(0, str(gr_limit))

# Checkbuttons
masking_var = tk.BooleanVar(value=True)
masking_check = tk.Checkbutton(root, variable=masking_var)
masking_check.grid(row=6, column=1)

gr_filter_var = tk.BooleanVar(value=False)
gr_filter_check = tk.Checkbutton(root, variable=gr_filter_var)
gr_filter_check.grid(row=7, column=1)

# Process button
process_button = tk.Button(root, text="Process LAS Files", command=process_las_files)
process_button.grid(row=8, column=1, columnspan=3, pady=10)

# Simple Scoping Log Viewer button
simple_viewer_button = tk.Button(root, text="Simple Scoping Log Viewer", command=simple_scoping_viewer)
simple_viewer_button.grid(row=9, column=1, columnspan=3, pady=10)

# Scoping Log Viewer button
scoping_viewer_button = tk.Button(root, text="Scoping Log Viewer", command=scoping_viewer)
scoping_viewer_button.grid(row=10, column=1, columnspan=3, pady=10)

root.mainloop()
