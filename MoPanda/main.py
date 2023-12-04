import os
import tkinter as tk
from tkinter import ttk

from modules import wellfilter


# Function to create a GUI with tabs
def create_main_gui():
    def open_copy_tab():
        from modules.utils import copyfiles
        copyfiles()

    def open_scoping_tab():
        from modules.scoping import scoping
        masking_var = tk.BooleanVar()
        gr_filter_var = tk.BooleanVar(value=False)
        input_folder_var = tk.StringVar(value=os.getcwd())  # StringVar to store the selected input folder path
        scoping(input_folder_var, masking_var, gr_filter_var)

    def open_dlis_viewer_tab():
        from modules.dlis_io import dlis_viewer
        dlis_viewer()

    def open_las_viewer_tab():
        # Importing the WellLogGUI class here to avoid circular imports
        from modules.las_viewer import WellLogGUI
        las_viewer = WellLogGUI(app)

    def open_curve_fitter():
        from modules.curvefitter import CMRPermeabilityApp
        CMRPermeabilityApp()

    def open_log_calculator_tab():
        from modules.log_calculator import LogCalculator
        LogCalculator()

    def open_inwellpredictor():
        from modules.inwellpredictor import InWellPredictor
        InWellPredictor()

    def open_crosswellpredictor():
        from modules.crosswellpredictor import CrossWellPredictor
        CrossWellPredictor()

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
    ttk.Label(pp_tab, text="Log Calculator and Viewer", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
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

    # === Geomechanics Tab ===
    gm_tab = ttk.Frame(tab_control)
    tab_control.add(gm_tab, text="Geomechanics")

    # --- Log Predictor section ---
    ttk.Label(gm_tab, text="Stress", font=label_font).pack(anchor=tk.W, padx=10, pady=5)
    add_separator(gm_tab)

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

    tab_control.pack(expand=1, fill="both")
    app.mainloop()


if __name__ == "__main__":
    create_main_gui()