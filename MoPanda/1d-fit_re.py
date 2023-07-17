import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import pandas as pd

# Load the T2 distribution dataset from the Excel spreadsheet
input_file_path = './data/cmr/Denova1_MD.xlsx'
df = pd.read_excel(input_file_path)
component = 4
# Extract the depth column and T2 distribution data
depths = df.iloc[:, 0].values
t2_data = df.iloc[:, 1:].values

ls = np.linspace(np.log10(0.3), np.log10(3000), len(t2_data[0]))  # Generate logarithmically spaced T2 values

# Select a T2 distribution for fitting
distribution = t2_data[1077]

# Define a model function with multiple Gaussian components
def multi_gauss(x, *params):
    result = np.zeros_like(x)
    num_components = len(params) // 3
    for i in range(num_components):
        A, mu, sigma = params[i * 3: (i + 1) * 3]
        result += A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))
    return result

# Perform automatic searching
auto_search = False  # Set to False if manual mode is preferred
min_components = 1
max_components = 5

if auto_search:
    # Define a function to evaluate the AIC for a given number of components
    def evaluate_aic(x, distribution, num_components):
        p0 = [0.] * (3 * num_components)
        coeff, _ = curve_fit(multi_gauss, x, distribution, p0=p0)
        residuals = distribution - multi_gauss(x, *coeff)
        mse = np.mean(residuals ** 2)
        num_params = 3 * num_components
        n = len(x)
        aic = n * np.log(mse) + 2 * num_params
        return aic

    aic_values = []
    for n in range(min_components, max_components + 1):
        aic = evaluate_aic(ls, distribution, n)
        aic_values.append(aic)

    best_n_components = np.argmin(aic_values) + min_components

    # Perform curve fitting with the best number of components
    p0 = [0.] * (3 * best_n_components)
    coeff, var_matrix = curve_fit(multi_gauss, ls, distribution, p0=p0)
    num_components = best_n_components
else:
    # Prompt the user for the number of components
    num_components = component
    # num_components = int(input("Enter the number of Gaussian components: "))

    # Perform curve fitting with the best number of components
    p0 = [1./num_components, (np.log10(3000)/num_components), 1.] * num_components
    bounds = ([0., np.log10(0.3), 1e-6] * num_components, [1., np.log10(3000), 1.5] * num_components)
    coeff, var_matrix = curve_fit(multi_gauss, ls, distribution, p0=p0, bounds=bounds)


# Generate curves for each Gaussian component
gauss_components = []
for i in range(num_components):
    pg = coeff[i * 3: (i + 1) * 3]
    gauss_components.append(multi_gauss(ls, *pg))

# Plot the data and fitted Gaussian components
plt.semilogx(10**ls, distribution, label='Data')
for i, component in enumerate(gauss_components):
    plt.semilogx(10**ls, component, label=f'Gaussian {i+1}')

plt.legend()
plt.show()
