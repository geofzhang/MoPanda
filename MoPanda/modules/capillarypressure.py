import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tkinter import filedialog, messagebox, Tk, Label, Button, Entry, Text
import scienceplots

plt.style.use(['science'])


# Define the equation to fit the data
def fit_function(X, a, Pe):
    return Pe * np.power(X, -1 / a)


def krw(X, a):
    return np.power(X, (2 + a) / a)


def krg(X, a):
    return ((1 - X) ** 2) * (1 - np.power(X, (2 + a) / a))


def swi(sw, swr, srg, snwr):
    sw = sw * (1 - snwr) + snwr
    return (sw - swr) / (1 - swr - srg)


class RelPerm:
    def __init__(self, root):
        self.root = root
        root.title("Relative Permeability Calculator")
        self.max_cp = 41  # in psi
        self.srg = 0.3  # trapped/residual gas saturation
        self.pe_ratio = None

        # GUI part of the init
        self.filename = ""
        self.textbox = Text(root, wrap="none", height=1, width=40)
        self.textbox.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.load_button = Button(root, text="Load MICP Data File", command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=10, pady=10)

        self.param_textbox = Text(root, wrap="none", height=1, width=40)
        self.param_textbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.param_load_button = Button(root, text="Load MICP Parameter File", command=lambda: self.load_file('param'))
        self.param_load_button.grid(row=1, column=2, padx=10, pady=10)

        self.pe_lab_label = Label(root, text=r"Enter the Lab Entry Pressure Pe_lab (psi):")
        self.pe_lab_label.grid(row=2, column=0, padx=10, pady=10)
        self.pe_lab_entry = Entry(root)
        self.pe_lab_entry.grid(row=2, column=2, padx=10, pady=10)

        self.pe_res_label = Label(root, text=r"Enter the Reservoir Entry Pressure Pe_res (psi):")
        self.pe_res_label.grid(row=3, column=0, padx=10, pady=10)
        self.pe_res_entry = Entry(root)
        self.pe_res_entry.grid(row=3, column=2, padx=10, pady=10)

        self.cp_label = Label(root, text=r"Enter the Capillary Pressure Cap (psi):")
        self.cp_label.grid(row=4, column=0, padx=10, pady=10)
        self.cp_entry = Entry(root)
        self.cp_entry.grid(row=4, column=2, padx=10, pady=10)
        self.cp_entry.insert(0, "41")  # Inserting default value

        # Label and Entry for Trapped CO2 Saturation
        self.srg_label = Label(root, text=r"Enter the Trapped CO2 Saturation:")
        self.srg_label.grid(row=5, column=0, padx=10, pady=10)
        self.srg_entry = Entry(root)
        self.srg_entry.grid(row=5, column=2, padx=10, pady=10)
        self.srg_entry.insert(0, "0.3")  # Inserting default value

        self.submit_button = Button(root, text="Submit", command=self.process_file)
        self.submit_button.grid(row=6, column=2, columnspan=1, padx=10, pady=10)

    def update_entries(self):
        if hasattr(self, 'param_filename') and hasattr(self, 'sample_name'):
            # Load the parameter file and update the entries
            param_df = pd.read_excel(self.param_filename)
            params = param_df[param_df['Sample'] == self.sample_name]
            if not params.empty:
                self.pe_lab_entry.delete(0, 'end')  # Clear previous content
                self.pe_lab_entry.insert(0, str(params['Pe_lab'].values[0]))  # Insert Pe_lab
                self.pe_res_entry.delete(0, 'end')  # Clear previous content
                self.pe_res_entry.insert(0, str(params['Pe_res'].values[0]))  # Insert Pe_res
            else:
                print(f"No parameters found for sample {self.sample_name}")
                # Show error message if no parameters are found
                messagebox.showerror("Error", f"No parameters found for sample {self.sample_name}, "
                                              f"please input Pe_lab and Pe_res manually.")

    def load_file(self, filetype='data'):
        initial_dir = "../data/core/Rel-k"
        filepath = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select file",
            filetypes=(("excel files", "*.xlsx"), ("all files", "*.*"))
        )
        if filepath:
            filename = os.path.basename(filepath)
            if filetype == 'data':
                self.filename = filepath
                self.textbox.delete("1.0", "end")
                self.textbox.insert("1.0", filename)

                name, ext = os.path.splitext(filename)
                sample_name = name.split()[-1]
                self.sample_name = sample_name
                self.update_entries()  # Call update_entries after setting sample_name
            elif filetype == 'param':
                self.param_filename = filepath
                self.param_textbox.delete("1.0", "end")
                self.param_textbox.insert("1.0", filename)
                self.update_entries()  # Call update_entries after setting param_filename

    def process_file(self):
        try:
            if not self.filename:
                raise Exception("No file loaded.")

            Pe = float(self.pe_res_entry.get()) * 0.00689476
            self.pe_ratio = float(self.pe_res_entry.get()) / float(self.pe_lab_entry.get())
            # print(self.pe_ratio)

        except Exception as e:
            # Handle or display the error message
            print(f"An error occurred: {e}")

        # Updating self.max_cp and self.srg from user input
        self.max_cp = float(self.cp_entry.get())
        self.srg = float(self.srg_entry.get())

        df = pd.read_excel(self.filename)
        print(df)
        df['Capillary Pressure (MPa)'] = df['Capillary Pressure (psi)'] * 0.00689476
        max_cp_mpa = self.max_cp * 0.00689476
        snwr = np.min(df['Pseudo Wetting-phase Saturation'])
        df['Capillary Pressure_Reservoir (MPa)'] = df['Capillary Pressure (MPa)'] * self.pe_ratio
        df['Normalized Wetting-phase Saturation'] = (df['Pseudo Wetting-phase Saturation'] - snwr) / (1 - snwr)
        Y = df['Capillary Pressure_Reservoir (MPa)']
        X = df['Normalized Wetting-phase Saturation']
        popt, _ = curve_fit(fit_function, X, Y)
        a = popt[0]
        print(a)
        sw_min = df['Pseudo Wetting-phase Saturation'][df['Capillary Pressure (MPa)'] <= max_cp_mpa].iloc[-1]

        Xd = np.linspace(1, sw_min, 50)
        Xi = swi(Xd, sw_min, self.srg, snwr)

        new_df = pd.DataFrame({'Sw_d': Xd})
        new_df['Sw_i'] = Xi
        new_df['krw_d'] = krw(Xd, a)
        new_df['krg_d'] = krg(Xd, a)
        new_df['krw_i'] = krw(Xi, a)
        new_df['krg_i'] = krg(Xi, a)
        new_df['lambda'] = a

        # Save the original DataFrame to one sheet and new_df to another sheet in the same Excel file
        with pd.ExcelWriter(self.filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Original_Data', index=False)
            new_df.to_excel(writer, sheet_name='Processed_Data', index=False)

        # Plot the curves
        fig, ax1 = plt.subplots(figsize=(10, 6))

        line1, = ax1.plot(Xd, new_df['krw_d'], label='krw_drainage', color='#0C5DA5', ls='-')
        line2, = ax1.plot(Xd, new_df['krg_d'], label='krg_drainage', color='#FF2C00', ls='-')
        line3, = ax1.plot(Xi, new_df['krw_i'], label='krw_imbibition', color='#0C5DA5', ls='--')
        line4, = ax1.plot(Xi, new_df['krg_i'], label='krg_imbibition', color='#FF2C00', ls='--')

        ax1.set_xlabel('Water Saturation')
        ax1.set_ylabel('Relative Permeability')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()  # Create a second y-axis
        line5, = ax2.plot(X, Y, color='#00B945', ls='-',
                          label='Capillary Pressure\t')  # Plotting on the secondary y-axis
        ax2.set_ylabel(r'Capillary Pressure ($MPa$)')

        # Combine legends from both axes
        lines = [line1, line2, line3, line4, line5]
        labels = [l.get_label() for l in lines]

        ax2.legend(lines, labels, loc=0)
        plt.title(f'Relative Permeability Curves for Sample {self.sample_name}')
        plt.show()

root = Tk()
my_app = RelPerm(root)
root.mainloop()
