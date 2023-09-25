import os
import shutil
import tkinter as tk
import xml.etree.ElementTree as ET
from tkinter import filedialog

import pandas as pd


def check_file(output_file):
    if os.path.exists(output_file):
        while True:
            response = input(
                f"The output file '{output_file}' already exists. Do you want to overwrite it? (Y/N): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                return False
            else:
                print("Invalid response. Please enter 'Y' or 'N'.")
    return True


class ColorCoding:
    def __init__(self):
        self.df = pd.DataFrame(columns=['name', 'label', 'color'])

    def litho_color(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        data = []
        for lithology in root.findall('lithology'):
            name = lithology.get('name')
            label = int(lithology.get('label'))
            color = lithology.get('color')

            data.append({'name': name, 'label': label, 'color': color})

        self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

        return self.df

    def name_to_label(self, lithology_name):
        if isinstance(lithology_name, (list, tuple)):
            return self.df.loc[self.df['name'].isin(lithology_name), 'label'].tolist()
        else:
            label = self.df.loc[self.df['name'] == lithology_name, 'label']
            return label.values[0] if not label.empty else None

    def label_to_name(self, lithology_label):
        if isinstance(lithology_label, (list, tuple)):
            return self.df.loc[self.df['label'].isin(lithology_label), 'name'].tolist()
        else:
            name = self.df.loc[self.df['label'] == lithology_label, 'name']
            return name.values[0] if not name.empty else None

    def label_to_color(self, lithology_label):
        if isinstance(lithology_label, (list, tuple)):
            return self.df.loc[self.df['label'].isin(lithology_label), 'color'].tolist()
        else:
            color = self.df.loc[self.df['label'] == lithology_label, 'color']
            return color.values[0] if not color.empty else None

    def name_to_color(self, lithology_name):
        if isinstance(lithology_name, (list, tuple)):
            return self.df.loc[self.df['name'].isin(lithology_name), 'color'].tolist()
        else:
            color = self.df.loc[self.df['name'] == lithology_name, 'color']
            return color.values[0] if not color.empty else None


def update_columns(log, column_a, column_b, method='left'):
    """
    Updates column_a with not-null values from column_b and vice versa according to the specified method.

    Arguments:
    log -- pandas DataFrame.
    column_a, column_b -- names of the columns in the DataFrame.
    method -- a string indicating the updating method: 'left', 'right', or 'mean' (default 'left').
            'left' keeps the column_a and only update not-null values from column_b
            'right' keeps the column_b and only update not-null values from column_a
            'mean' keeps the column_a and only update not-null values from column_b

    Returns:
    Updated pandas Series according to the updating method.
    """
    if column_a not in log.columns or column_b not in log.columns:
        raise ValueError(f"Columns {column_a} and/or {column_b} not found in DataFrame.")

    if method == 'left':
        # Update column_a with not null values of column_b
        log[column_a].update(log[column_b])
        return log[column_a]
    elif method == 'right':
        # Update column_b with not null values of column_a
        log[column_b].update(log[column_a])
        return log[column_b]
    elif method == 'mean':
        # Return mean of column_a and column_b, ignoring NA
        return log[[column_a, column_b]].mean(axis=1)
    else:
        raise ValueError("Method must be one of 'left', 'right', or 'mean'.")


def copy_files_with_extension(source_folder, destination_folder, file_extension):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(file_extension):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_folder)
                destination_path = os.path.join(destination_folder, relative_path)

                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copyfile(source_path, destination_path)
                print(f"Copied {source_path} to {destination_path}")


def browse_input_folder(input_folder_entry):
    input_folder = filedialog.askdirectory()
    input_folder_entry.delete(0, tk.END)
    input_folder_entry.insert(0, input_folder)


def browse_output_folder(output_folder_entry):
    output_folder = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, output_folder)


def start_copying(input_folder_entry, output_folder_entry, file_extension_entry):
    input_folder = input_folder_entry.get()
    output_folder = output_folder_entry.get()
    file_extension = file_extension_entry.get()

    if input_folder and output_folder and file_extension:
        copy_files_with_extension(input_folder, output_folder, file_extension)
        return "Copying completed successfully!"
    else:
        return "Please provide all the required information."


def copyfiles():
    app = tk.Tk()
    app.title("File Copy With Extension")

    app.geometry("800x500")

    input_folder_label = tk.Label(app, text="Select Input Folder:")
    input_folder_label.pack()
    input_folder_entry = tk.Entry(app, width=50)
    input_folder_entry.pack()
    input_folder_button = tk.Button(app, text="Browse", command=lambda: browse_input_folder(input_folder_entry))
    input_folder_button.pack()

    output_folder_label = tk.Label(app, text="Select Output Folder:")
    output_folder_label.pack()
    output_folder_entry = tk.Entry(app, width=50)
    output_folder_entry.pack()
    output_folder_button = tk.Button(app, text="Browse", command=lambda: browse_output_folder(output_folder_entry))
    output_folder_button.pack()

    file_extension_label = tk.Label(app, text="Enter File Extension (e.g., .las):")
    file_extension_label.pack()
    file_extension_entry = tk.Entry(app, width=10)
    file_extension_entry.pack()

    start_button = tk.Button(app, text="Start Copying", command=lambda: print(
        start_copying(input_folder_entry, output_folder_entry, file_extension_entry)))
    start_button.pack()

    app.mainloop()
