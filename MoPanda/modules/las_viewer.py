import os
import tkinter as tk
from tkinter import filedialog, messagebox
from modules.las_io import LasIO


def resize_image(image, size_limit):
    width, height = image.width(), image.height()
    if width > size_limit or height > size_limit:
        if width >= height:
            new_width = size_limit
            new_height = int((size_limit / width) * height)
        else:
            new_height = size_limit
            new_width = int((size_limit / height) * width)
        return image.subsample(width // new_width, height // new_height)
    else:
        return image


class WellLogGUI(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("LAS Viewer")
        self.geometry("800x800")

        # Initialize a list to store LAS files and their associated data
        self.las_files = []

        # Initialize the output path
        self.output_path = "../output"

        self.create_widgets()

    def create_widgets(self):
        # Buttons in the first column
        self.select_file_button = tk.Button(self, text="Select LAS Files", command=self.select_las_files)
        self.select_file_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Load the help icon image and resize it
        help_icon = tk.PhotoImage(file="./data/images/help_small.png")
        # help_icon = resize_image(help_icon, 5)  # Limit the icon size to 5

        # Create a label for the help icon
        self.help_label = tk.Label(self, image=help_icon, cursor="question_arrow")
        self.help_label.image = help_icon  # Keep a reference to prevent garbage collection
        self.help_label.grid(row=0, column=2, padx=10, pady=10)

        # Tooltip for the help icon
        self.create_tooltip(self.help_label, "1. Support selection of single or multiple .las files;\n"
                                             "2. Select the desired UWI to view the specific well information with the buttons below;\n"
                                             "3. All wells will be exported together.")

        self.display_well_button = tk.Button(self, text="Display Well", command=self.display_well)
        self.display_well_button.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.display_params_button = tk.Button(self, text="Display Parameters", command=self.display_params)
        self.display_params_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.display_curves_button = tk.Button(self, text="Display Curves", command=self.display_curves)
        self.display_curves_button.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.select_output_button = tk.Button(self, text="Select Output Path", command=self.select_output_path)
        self.select_output_button.grid(row=4, column=0, padx=10, pady=5, sticky="w")

        self.export_button = tk.Button(self, text="Export Data", command=self.export_data)
        self.export_button.grid(row=5, column=0, padx=10, pady=5, sticky="w")

        # Dropdown menu in the second column, same row as select_file_button
        self.selected_file_var = tk.StringVar()
        self.file_dropdown = tk.OptionMenu(self, self.selected_file_var, "")
        self.file_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Text area to display the selected output path
        self.output_text = tk.Text(self, width=60, height=1)
        self.output_text.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Text area to display information
        self.text_well = tk.Text(self, width=97, height=33)
        self.text_well.grid(row=6, column=0, columnspan=3, padx=10, pady=5, sticky="sew")

    # Resize an image while maintaining the original aspect ratio

    def select_las_files(self):
        # Allow the user to select multiple LAS files
        file_paths = filedialog.askopenfilenames(filetypes=[("LAS Files", "*.las")])

        if file_paths:
            # Create LasIO objects for each selected LAS file
            self.las_files = [LasIO(file_path) for file_path in file_paths]

            # Extract filenames without extensions
            file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

            # Update the file dropdown menu with the selected file names (without extensions)
            self.file_dropdown['menu'].delete(0, 'end')  # Clear the existing menu items
            for file_name in file_names:
                self.file_dropdown['menu'].add_command(label=file_name,
                                                       command=tk._setit(self.selected_file_var, file_name))
            self.selected_file_var.set(file_names[0])  # Select the first file by default

            messagebox.showinfo("Logs Loaded", f"{len(file_paths)} logs loaded successfully.")

    # Create a tooltip function to display instructions
    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25

            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = tk.Label(self.tooltip, text=text, background="lightyellow", relief="solid", borderwidth=1)
            label.pack()

        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

    def display_well(self):
        selected_file_name = self.selected_file_var.get()
        for las_file in self.las_files:
            if las_file.filename == selected_file_name:  # Use the filename attribute
                well_info = str(las_file.well)  # Replace with the appropriate method in LasIO
                self.text_well.delete("1.0", tk.END)
                self.text_well.insert(tk.END, well_info)
                self.update()
                return

    def display_params(self):
        selected_file_name = self.selected_file_var.get()
        for las_file in self.las_files:
            if las_file.filename == selected_file_name:  # Use the filename attribute
                params_info = str(las_file.params)  # Replace with the appropriate method in LasIO
                self.text_well.delete("1.0", tk.END)
                self.text_well.insert(tk.END, params_info)
                self.update()
                return

    def display_curves(self):
        selected_file_name = self.selected_file_var.get()
        for las_file in self.las_files:
            if las_file.filename == selected_file_name:  # Use the filename attribute
                curves_info = str(las_file.curves)  # Replace with the appropriate method in LasIO
                self.text_well.delete("1.0", tk.END)
                self.text_well.insert(tk.END, curves_info)
                self.update()
                return

    def select_output_path(self):
        self.output_path = filedialog.askdirectory()
        self.output_text.delete("1.0", tk.END)  # Clear existing text
        self.output_text.insert(tk.END, self.output_path)

    def export_data(self):
        if self.output_path:
            for las_file in self.las_files:
                filename = os.path.splitext(las_file.filename)[0]
                csv_filename = filename + "_las2csv.csv"
                excel_filename = filename + "_las2excel.xlsx"
                csv_path = os.path.join(self.output_path, csv_filename)
                excel_path = os.path.join(self.output_path, excel_filename)

                las_file.export_csv(csv_path)
                las_file.export_excel(excel_path)

            messagebox.showinfo("Export Successful", "Data exported successfully for all LAS files.")
        else:
            messagebox.showerror("Export Error", "No output path selected.")


if __name__ == "__main__":
    app = WellLogGUI()
    app.mainloop()
