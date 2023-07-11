import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

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
    channels_window.title('Channels Data')
    channels_text = tk.Text(channels_window)
    channels_text.pack(expand=True, fill='both')
    channels_text.insert(tk.END, channels.to_string(index=True))

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
                    messagebox.showinfo('Channel Data', str(channel_data))
            except ValueError:
                show_error_message('Invalid index number.')
        elif input_name:
            try:
                index = channels[channels['Name'] == input_name].index[0]
                channel_data = f.channels[index].curves()
                messagebox.showinfo('Channel Data', str(channel_data))
            except IndexError:
                show_error_message('Channel with the given name not found.')
        else:
            show_error_message('Please enter an index number or name.')

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

def show_error_message(message):
    messagebox.showerror('Error', message)

root = tk.Tk()

load_button = tk.Button(root, text='Load DLIS File', command=load_dlis_file)
load_button.pack(pady=10)

root.mainloop()
