from data_analysis import fit_curve

# Example usage:
csv_file = '../data/Denova1_core_k.csv'
x1_column = 'BVI'
y1_column = 'T2lm_BVI'
x2_column = 'FFI'
y2_column = 'T2lm_FFI'
z_column = 'Permeability_Core'

a1_fit, m1_fit, n1_fit, a2_fit, m2_fit, n2_fit, rmse = fit_curve(csv_file, x1_column, y1_column, x2_column, y2_column,
                                                                 z_column)
