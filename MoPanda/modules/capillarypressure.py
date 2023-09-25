import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots
plt.style.use(['science'])

# Define the equation to fit the data
def fit_function(X, a, Pe):
    return Pe * np.power(X, -1 / a)


# Define functions for krw and krg

# Rel Perm for wetting phase (saline water in this case)
# Equation is based on Purcell relative permeability model
def krw(X, a):
    return np.power(X, (2 + a) / a)


# Rel Perm for nonwetting phase (gas or C02 in this case)
# Equation is based on Brooks-Corey relative permeability model
def krg(X, a):
    return ((1 - X) ** 2) * (1 - np.power(X, (2 + a) / a))


# Read the Excel file
excel_file = '../data/core/MICP.xlsx'
df = pd.read_excel(excel_file)

# Convert Capillary Pressure from psi to MPa
df['Capillary Pressure (MPa)'] = df['Capillary Pressure (psi)'] * 0.00689476

# Assign entry pressure Pe
Pe = 5.66

# Fit the data to the equation and get the parameter 'a'
Y = df['Capillary Pressure (MPa)']
X = df['Pseudo Wetting-phase Saturation']

# Perform the curve fitting
popt, _ = curve_fit(fit_function, X, Y)

# 'a' parameter is the first element of the popt array
a = popt[0]

# Calculate krw and krnw using 'a'
df['krw'] = krw(X, a)
df['krg'] = krg(X, a)

# Write the krw and krnw columns back to the original Excel file
df.to_excel(excel_file, index=False)

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(X, df['krw'], label='krw')
plt.plot(X, df['krg'], label='krg')
# plt.plot(X, df['krw'], label='krw', linestyle='-', color='blue')
# plt.plot(X, df['krg'], label='krg', linestyle='--', color='red')
plt.xlabel('Water Saturation')
plt.ylabel('Relative Permeability')
plt.xlim(1, 0)  # Limit x-axis from 1 to 0
plt.ylim(0, 1)  # Limit y-axis from 0 to 1
plt.legend()
plt.title('Relative Permeability Curves')
plt.grid(False)
plt.show()
