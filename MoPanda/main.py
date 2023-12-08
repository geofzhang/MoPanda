import os
import tkinter as tk
from tkinter import ttk
import sys
import threading

from PyQt6.QtWidgets import QApplication

from modules import wellfilter


class LASEditorThread(threading.Thread):
    # Testing: Create new thread for las_editor to avoid potential conflict between QT and TK framework.
    def __init__(self):
        threading.Thread.__init__(self)
        self.app = None

    def run(self):
        self.app = QApplication(sys.argv)
        from modules.las_editor import LASEditor
        editor = LASEditor()
        editor.show()
        self.app.exec()

    def stop(self):
        if self.app:
            self.app.quit()


# Function to create a GUI with tabs
def create_main_gui():
    def open_copy_tab():
        from modules.utils import copyfiles
        copyfiles()

    def open_scoping_tab():
        from modules.scoping import MaskingForScoping
        scoping_window = tk.Toplevel()
        masking_var = tk.BooleanVar()
        gr_filter_var = tk.BooleanVar(value=False)
        input_folder_var = tk.StringVar(value=os.getcwd())  # StringVar to store the selected input folder path
        MaskingForScoping(scoping_window, input_folder_var, masking_var, gr_filter_var)

    def open_dlis_viewer_tab():
        from modules.dlis_io import dlis_viewer
        dlis_viewer()

    def open_las_viewer_tab():
        # Importing the WellLogGUI class here to avoid circular imports
        from modules.las_viewer import WellLogGUI
        WellLogGUI(app)

    def open_geomechanics():
        pass

    def open_relk():
        from modules.relk import RelPerm
        RelPerm(app)

    def open_cp():
        pass

    def open_las_editor():
        # las_editor_thread = LASEditorThread()
        # las_editor_thread.start()

        app = QApplication(sys.argv)
        from modules.las_editor import LASEditor
        editor = LASEditor()
        app.exec()

    def open_inwellpredictor():
        from modules.inwellpredictor import InWellPredictor
        InWellPredictor(app)

    def open_crosswellpredictor():
        from modules.crosswellpredictor import CrossWellPredictor
        CrossWellPredictor()

    def open_legacy_log_converter():
        from modules.legacylog import LegacyLogConverter
        converter = LegacyLogConverter()
        converter.run()

    # Define custom font for labels
    label_font = ("Arial", 10, "bold")  # Change "Arial" to any other font if desired.

    # Main application window
    app = tk.Tk()
    app.title("MoPanda - Modular Petrophysics and Data Analysis Tool")

    # Set the window size
    app.geometry("800x800")

    # Create a tab manager
    tab_control = ttk.Notebook(app)

    # === Data Management Tab ===
    dm_tab = ttk.Frame(tab_control)
    tab_control.add(dm_tab, text="Data Management")

    # Separator function for convenience
    def add_separator(parent):
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.BOTH, padx=10, pady=10)

    # --- Well Scraper section ---
    ttk.Label(dm_tab, text="Well Scraper", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(dm_tab)
    # Placeholder for future use
    ttk.Label(dm_tab, text="Placeholder for future content").pack(anchor=tk.W, padx=20, pady=5)
    add_separator(dm_tab)

    # --- File Management section ---
    ttk.Label(dm_tab, text="File Management", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(dm_tab)
    # File Copying
    open_copy_button = tk.Button(dm_tab, text="Open File Copying", command=open_copy_tab)
    open_copy_button.pack(anchor=tk.W, padx=20, pady=5)
    # Well Filter
    open_well_filter_button = tk.Button(dm_tab, text="Open Well Filter", command=wellfilter.open_well_filter_window)
    open_well_filter_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(dm_tab)

    # --- Log Reader section ---
    ttk.Label(dm_tab, text="Log Reader", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(dm_tab)
    # DLIS Viewer
    open_dlis_viewer_button = tk.Button(dm_tab, text="Open DLIS Viewer", command=open_dlis_viewer_tab)
    open_dlis_viewer_button.pack(anchor=tk.W, padx=20, pady=5)
    # LAS Viewer
    open_las_viewer_button = tk.Button(dm_tab, text="Open LAS Viewer", command=open_las_viewer_tab)
    open_las_viewer_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(dm_tab)

    # === Edit & Display ===
    edit_tab = ttk.Frame(tab_control)
    tab_control.add(edit_tab, text="Edit & Display")

    # --- LAS editor section ---
    ttk.Label(edit_tab, text="LAS Edit & Display", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(edit_tab)

    # LAS editor
    open_laseditor_button = tk.Button(edit_tab, text="Open Interactive LAS Editor and Displayer",
                                      command=open_las_editor)
    open_laseditor_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(edit_tab)

    # === Data Analysis Tab ===
    da_tab = ttk.Frame(tab_control)
    tab_control.add(da_tab, text="Data Analysis")

    # --- Explorative Data Analysis section ---
    ttk.Label(da_tab, text="Explorative Data Analysis", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(da_tab)
    # Placeholder for future use
    ttk.Label(da_tab, text="Placeholder for future content").pack(anchor=tk.W, padx=20, pady=5)
    add_separator(da_tab)

    # --- Curve Fitter section ---
    ttk.Label(da_tab, text="Permeability Model Fitting", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(da_tab)
    #  Permeability Model Fitting
    cmr_frame = ttk.Frame(da_tab)
    cmr_frame.pack(fill="both", padx=20, pady=20)
    from modules.curvefitter import CMRPermeabilityApp
    CMRPermeabilityApp(cmr_frame)
    add_separator(da_tab)

    # --- Log Predictor section ---
    ttk.Label(da_tab, text="Log Predictor", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(da_tab)
    # In Well Log Predictor
    open_inwellpredictor_button = tk.Button(da_tab, text="In Well Log Predictor", command=open_inwellpredictor)
    open_inwellpredictor_button.pack(anchor=tk.W, padx=20, pady=5)
    # Cross Well Log Predictor
    open_crosswellpredictor_button = tk.Button(da_tab, text="Cross Well Log Predictor", command=open_crosswellpredictor)
    open_crosswellpredictor_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(da_tab)

    # === Petrophysics Tab ===
    pp_tab = ttk.Frame(tab_control)
    tab_control.add(pp_tab, text="Petrophysics")

    # --- Log Calculator section ---
    ttk.Label(pp_tab, text="Petrophysical Analysis", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(pp_tab)
    log_calculator_frame = ttk.Frame(pp_tab)
    log_calculator_frame.pack(fill="both", padx=10, pady=5)
    # Embed LogCalculator into the Petrophysics tab
    from modules.log_calculator import LogCalculator
    LogCalculator(log_calculator_frame)
    add_separator(pp_tab)

    # --- CMR section ---
    ttk.Label(pp_tab, text="CMR", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(pp_tab)

    # === Constitutive RelationTab ===
    cr_tab = ttk.Frame(tab_control)
    tab_control.add(cr_tab, text="Constitutive")

    # --- Geomechanics section ---
    ttk.Label(cr_tab, text="Geomechanics", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(cr_tab)

    # Geomechanics
    open_geomechanics_button = tk.Button(cr_tab, text="Geomechanics", command=open_geomechanics)
    open_geomechanics_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(cr_tab)

    # --- relative permeability section ---
    ttk.Label(cr_tab, text="Relative Permeability", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(cr_tab)

    # relk modeling
    open_relk_button = tk.Button(cr_tab, text="Rel-k modeling", command=open_relk)
    open_relk_button.pack(anchor=tk.W, padx=20, pady=5)

    # Capillary pressure
    open_cp_button = tk.Button(cr_tab, text="Capillary pressure modeling", command=open_cp)
    open_cp_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(cr_tab)

    # === Regional Analysis Tab ===
    ra_tab = ttk.Frame(tab_control)
    tab_control.add(ra_tab, text="Regional Analysis")

    # Scoping Viewer
    open_viewer_button = tk.Button(ra_tab, text="Open Scoping Module", command=open_scoping_tab)
    open_viewer_button.pack(pady=20)

    # === Others Tab ===
    others_tab = ttk.Frame(tab_control)
    tab_control.add(others_tab, text="Others")
    # Note: Implement any content for 'Others' here when needed

    # --- core analysis section ---
    ttk.Label(others_tab, text="Core Analysis", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(others_tab)

    # --- core analysis section ---
    ttk.Label(others_tab, text="Other Log Calculations", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(others_tab)
    # Capillary pressure
    open_legacy_button = tk.Button(others_tab, text="Legacy Log Converter", command=open_legacy_log_converter)
    open_legacy_button.pack(anchor=tk.W, padx=20, pady=5)
    add_separator(others_tab)

    tab_control.pack(expand=1, fill="both")
    app.mainloop()


if __name__ == "__main__":
    create_main_gui()
