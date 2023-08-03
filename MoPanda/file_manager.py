import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from EDA.MoPanda.MoPanda.modules.las_io import LasIO
import os
import shutil
import xml.etree.ElementTree as ET


def get_root_names():
    file_dir = os.path.dirname(__file__)
    alias_path = os.path.join(file_dir, 'data/log_info', 'log_alias.xml')

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


# Function to create a GUI with tabs
def create_main_gui():
    def open_copy_tab():
        from EDA.MoPanda.MoPanda.modules.utils import copyfiles
        copyfiles()

    def open_scoping_tab():
        from EDA.MoPanda.MoPanda.modules.scoping import scoping
        masking_var = tk.BooleanVar()
        gr_filter_var = tk.BooleanVar(value=False)
        input_folder_var = tk.StringVar(value=os.getcwd())  # StringVar to store the selected input folder path
        scoping(input_folder_var, masking_var, gr_filter_var)

    def open_dlis_viewer_tab():
        from EDA.MoPanda.MoPanda.modules.dlis_io import dlis_viewer
        dlis_viewer()

    def open_las_viewer_tab():
        # Importing the WellLogGUI class here to avoid circular imports
        from EDA.MoPanda.MoPanda.modules.las_viewer import WellLogGUI
        las_viewer = WellLogGUI(app)

    def open_log_calculator_tab():
        from EDA.MoPanda.MoPanda.modules.log_calculator import LogCalculator
        LogCalculator()

    def open_well_filter_tab():
        root_names = get_root_names()

        # You can pass the selected_root_names list to the check_alias_match() function

        # Well Filter Tab
        tab2 = ttk.Frame(tab_control)
        tab_control.add(tab2, text="Well Filter")
        tab_control.pack(expand=1, fill="both")

        # Checkbox Options
        checkboxes = []
        for root_name in root_names:
            var = tk.StringVar(value="")
            checkbox = tk.Checkbutton(tab2, text=root_name, variable=var, onvalue=root_name, offvalue="")
            checkbox.pack(anchor=tk.W)
            checkboxes.append(var)

        # # Apply Button
        # select_button = tk.Button(tab4, text="Select .las File", command=select_las_file)
        # select_button.pack(pady=10)

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
        single_button = tk.Button(tab2, text="Apply Filter to Single Well", command=apply_filter_single)
        single_button.pack(pady=10)

        # Apply Button to multiple wells
        multi_button = tk.Button(tab2, text="Apply Filter to Multiple Wells", command=apply_filter_multiple)
        multi_button.pack(pady=10)

    # Main application window
    app = tk.Tk()
    app.title("Main GUI")

    # Set the window size
    app.geometry("800x800")

    # Create a tab manager
    tab_control = ttk.Notebook(app)

    # Tab 1 - File Copying
    tab1 = ttk.Frame(tab_control)
    tab_control.add(tab1, text="File Copying")
    tab_control.pack(expand=1, fill="both")

    # Button to open the copy tab
    open_copy_button = tk.Button(tab1, text="Open File Copying", command=open_copy_tab)
    open_copy_button.pack(pady=20)

    # Tab 2 - Well Filter
    open_well_filter_tab()

    # Tab 3 - Scoping Viewer
    tab3 = ttk.Frame(tab_control)
    tab_control.add(tab3, text="Scoping")
    tab_control.pack(expand=1, fill="both")

    # Button to open the Scoping Viewer tab
    open_viewer_button = tk.Button(tab3, text="Open Scoping Module", command=open_scoping_tab)
    open_viewer_button.pack(pady=20)

    # Tab 4 - DLIS Viewer
    tab4 = ttk.Frame(tab_control)
    tab_control.add(tab4, text="DLIS Viewer")
    tab_control.pack(expand=1, fill="both")

    # Button to open the DLIS Viewer tab
    open_viewer_button = tk.Button(tab4, text="Open DLIS Viewer", command=open_dlis_viewer_tab)
    open_viewer_button.pack(pady=20)

    # Tab 5 - LAS Viewer
    tab5 = ttk.Frame(tab_control)
    tab_control.add(tab5, text="LAS Viewer")
    tab_control.pack(expand=1, fill="both")

    # Button to open the LAS Viewer tab
    open_las_viewer_button = tk.Button(tab5, text="Open LAS Viewer", command=open_las_viewer_tab)
    open_las_viewer_button.pack(pady=20)

    # Tab 6 - Log Calculator
    tab6 = ttk.Frame(tab_control)
    tab_control.add(tab6, text="Log Calculator")
    tab_control.pack(expand=1, fill="both")

    # Button to open the LAS Viewer tab
    open_las_viewer_button = tk.Button(tab6, text="Open Log Calculator", command=open_log_calculator_tab)
    open_las_viewer_button.pack(pady=20)

    # Start the main event loop
    app.mainloop()


if __name__ == "__main__":
    create_main_gui()
