import os
import shutil
import tkinter as tk
import xml.etree.ElementTree as ET
from tkinter import filedialog, messagebox

from modules.las_io import LasIO


def get_root_names():
    file_dir = os.path.dirname(__file__)
    alias_path = os.path.join(file_dir, '../data/log_info', 'log_alias.xml')

    if not os.path.isfile(alias_path):
        raise ValueError('No alias file at: %s' % alias_path)

    with open(alias_path, 'r') as f:
        root = ET.fromstring(f.read())

    root_names = [alias.tag for alias in root]
    return root_names


def select_las_file():
    las_file_path = filedialog.askopenfilename(filetypes=[("LAS Files", "*.las")])
    if las_file_path:
        log = LasIO(las_file_path)
        return log


def process_las_files(curves_list):
    """
    Process all .las files within the input folder, check for matching keys,
    and copy the matching files to the output folder while preserving the original structure.

    Parameters:
        input_folder (str): Path to the folder containing the .las files.
        output_folder (str): Path to the output folder where matching files will be copied.
        curves_list (list): A list of new curve names to check for matching keys.
    """
    input_folder = filedialog.askdirectory(title="Select Input Folder")
    output_folder = os.path.join(os.path.dirname(input_folder), 'Filtered wells', os.path.basename(input_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".las"):
                las_file_path = os.path.join(root, file)
                log = LasIO(las_file_path)

                if log.check_alias_match(curves_list):
                    # Create the destination folder preserving the original folder structure
                    relative_path = os.path.relpath(las_file_path, input_folder)
                    destination_folder = os.path.join(output_folder, os.path.dirname(relative_path))

                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)

                    # Copy the matching .las file to the destination folder
                    destination_file = os.path.join(destination_folder, file)
                    shutil.copy2(las_file_path, destination_file)


def open_well_filter_window():
    root_names = get_root_names()

    well_filter_window = tk.Toplevel()
    well_filter_window.title("Well Filter")

    # Checkbox Options
    checkboxes = []
    for root_name in root_names:
        var = tk.StringVar(value="")
        checkbox = tk.Checkbutton(well_filter_window, text=root_name, variable=var, onvalue=root_name, offvalue="")
        checkbox.pack(anchor=tk.W)
        checkboxes.append(var)

    def apply_filter_single():
        selected_root_names = [root_name.get() for root_name in checkboxes if root_name.get()]
        print("Selected Logs:", selected_root_names)
        log = select_las_file()
        if log.check_alias_match(selected_root_names):
            messagebox.showinfo('Log Filtering', f'Congrats! Well has all desired logs: {selected_root_names}.')
        else:
            messagebox.showerror('Error', 'Well is lacking at least one desired log.')

    def apply_filter_multiple():
        selected_root_names = [root_name.get() for root_name in checkboxes if root_name.get()]
        print("Selected Logs:", selected_root_names)
        process_las_files(selected_root_names)

    # Apply Button to single well
    single_button = tk.Button(well_filter_window, text="Apply Filter to Single Well", command=apply_filter_single)
    single_button.pack(pady=10)

    # Apply Button to multiple wells
    multi_button = tk.Button(well_filter_window, text="Apply Filter to Multiple Wells", command=apply_filter_multiple)
    multi_button.pack(pady=10)
