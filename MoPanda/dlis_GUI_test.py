import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import numpy as np
from dlisio import dlis


def load_dlis_file():
    filepath = filedialog.askopenfilename(filetypes=[('DLIS Files', '*.dlis')])
    if filepath:
        try:
            f, *tail = dlis.load(filepath)
            channels = summary_dataframe(f.channels, name='Name', long_name='Long Name',
                                         dimension='Dimension', units='Units', frame='Frame')
            show_channels_data(f, channels)
        except Exception as e:
            show_error_message(str(e))


def summary_dataframe(object, **kwargs):
    df = pd.DataFrame()
    for i, (key, value) in enumerate(kwargs.items()):
        list_of_values = []
        for item in object:
            try:
                x = getattr(item, key)
                list_of_values.append(x)
            except:
                list_of_values.append('')
                continue
        df[value] = list_of_values
    return df


def show_channels_data(f, channels):
    channels_window = tk.Toplevel(root)
    channels_window.geometry('1800x900')  # Set main window size to 1800x600
    channels_window.title('Channels Data')
    channels_text = tk.Text(channels_window)
    channels_text.pack(expand=True, fill='both')
    channels_text.insert(tk.END, channels.to_string(index=True))

    def measured_depth(channel_index):
        frame = f.channels[channel_index].frame
        condition = (channels['Name'] == 'TDEP') & (channels['Frame'] == frame)
        index = channels[condition].index[0] if condition.any() else None
        depth = f.channels[index].curves() if index is not None else None
        return depth

    def search_channels():
        search_query = search_entry.get().strip()
        if search_query:
            search_results = channels[
                (channels['Name'].str.contains(search_query, case=False)) |
                (channels['Long Name'].str.contains(search_query, case=False))
                ]
            channels_text.delete('1.0', tk.END)
            channels_text.insert(tk.END, search_results.to_string(index=True))

    search_frame = tk.Frame(channels_window)
    search_frame.pack(pady=10)

    search_label = tk.Label(search_frame, text='Search:')
    search_label.pack(side='left')
    search_entry = tk.Entry(search_frame)
    search_entry.pack(side='left')

    search_button = tk.Button(search_frame, text='Search', command=search_channels)
    search_button.pack(side='left')

    def get_channel_data():
        input_index = index_entry.get().strip()
        input_name = name_entry.get().strip()
        if input_index:
            try:
                index = int(input_index)
                if index < 0 or index >= len(channels):
                    show_error_message('Invalid index number.')
                else:
                    channel_data = f.channels[index].curves()
                    show_channel_data_and_stats(channel_data)
                    export_button['state'] = 'normal'  # Enable export button
                    export_button['command'] = lambda: export_data(channel_data, measured_depth(index))
            except ValueError:
                show_error_message('Invalid index number.')
        elif input_name:
            try:
                index = channels[channels['Name'] == input_name].index[0]
                if index < 0 or index >= len(channels):
                    show_error_message('Invalid channel name.')
                else:
                    channel_data = f.channels[index].curves()
                    show_channel_data_and_stats(channel_data)
                    export_button['state'] = 'normal'  # Enable export button
                    export_button['command'] = lambda: export_data(channel_data, measured_depth(index))
            except IndexError:
                show_error_message('Channel with the given name not found.')
        else:
            show_error_message('Please enter an index number or name.')

    def show_channel_data_and_stats(channel_data):
        data_stats_window = tk.Toplevel(root)
        data_stats_window.title('Channel Data and Statistics')

        # Create a frame for data and statistics
        frame = tk.Frame(data_stats_window)
        frame.pack(padx=10, pady=10)

        # Create a text widget to display channel data
        data_text = tk.Text(frame, height=20, width=80)
        data_text.insert(tk.END, str(channel_data))
        data_text.pack(side='left')

        # Check if channel_data is 1D or 2D
        if channel_data.ndim == 1:
            # Calculate statistics for 1D data
            min_value = np.min(channel_data)
            max_value = np.max(channel_data)
            mean_value = np.mean(channel_data)
            std_value = np.std(channel_data)
            interval = channel_data[1] - channel_data[0]
            num_measurements = len(channel_data)

            # Create labels to display statistics for 1D data
            stats_label = tk.Label(
                frame,
                text=f'Statistics:\nMin: {min_value:.2f}\nMax: {max_value:.2f}\nMean: {mean_value:.2f}\nStd: {std_value:.2f}\nInterval: {interval:.2f}\nNum Measurements: {num_measurements}'
            )
            stats_label.pack(side='right')
        elif channel_data.ndim == 2:
            # Calculate statistics for 2D data
            numeric_values = channel_data[np.isfinite(channel_data)]
            min_value = np.min(numeric_values)
            max_value = np.max(numeric_values)
            mean_value = np.mean(numeric_values)
            std_value = np.std(numeric_values)
            interval = channel_data[1, 0] - channel_data[0, 0]
            num_rows, num_cols = channel_data.shape

            # Create labels to display statistics for 2D data
            stats_label = tk.Label(
                frame,
                text=f'Statistics:\nMin: {min_value:.2f}\nMax: {max_value:.2f}\nMean: {mean_value:.2f}\nStd: {std_value:.2f}\nInterval: {interval:.2f}\nNum Rows: {num_rows}\nNum Columns: {num_cols}'
            )
            stats_label.pack(side='right')
        else:
            show_error_message('Unsupported data format.')

    def export_data(channel_data, depth):
        filepath = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[('Text Files', '*.txt'), ('CSV Files', '*.csv')]
        )
        if filepath:
            try:
                df = pd.DataFrame(channel_data)
                # Check if 'MD' column exists and move it to the first column
                if 'MD' in df.columns:
                    md_column = df['MD']
                    df = df.drop('MD', axis=1)
                    df.insert(0, 'MD', md_column)
                elif 'MD' not in df.columns:
                    df.insert(0, 'MD', depth * 0.1 / 12)  # Add empty 'MD' column as the first column:
                else:
                    with open(filepath, 'w') as file:
                        file.write(str(channel_data))

                # Reverse the DataFrame from Bottom-up to Top-down
                df = df[::-1]

                # Export the DataFrame as a CSV file
                df.to_csv(filepath, index=False)
                messagebox.showinfo('Export Successful', 'Channel data exported successfully.')
            except Exception as e:
                show_error_message(f'Failed to export channel data:\n{str(e)}')

    input_frame = tk.Frame(channels_window)
    input_frame.pack(pady=10)

    index_label = tk.Label(input_frame, text='Index:')
    index_label.pack(side='left')
    index_entry = tk.Entry(input_frame)
    index_entry.pack(side='left')

    name_label = tk.Label(input_frame, text='Name:')
    name_label.pack(side='left')
    name_entry = tk.Entry(input_frame)
    name_entry.pack(side='left')

    get_button = tk.Button(channels_window, text='Get Channel Data', command=get_channel_data)
    get_button.pack(pady=10)

    export_button = tk.Button(channels_window, text='Export Data', state='disabled')
    export_button.pack(pady=10)

    def close_window():
        channels_window.destroy()

    channels_window.protocol("WM_DELETE_WINDOW", close_window)


def show_error_message(message):
    messagebox.showerror('Error', message)


root = tk.Tk()
root.title("DLIS Viewer")

load_button = tk.Button(root, text='Load DLIS File', command=load_dlis_file)
load_button.pack(pady=10)

# Set window dimensions
window_width = 400
window_height = 100
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = int((screen_width / 2) - (window_width / 2))
y_position = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

root.mainloop()
