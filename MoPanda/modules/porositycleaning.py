import pandas as pd
import os


# Read the Excel file
input_file = 'Draco_xwell_predict.csv'  # Replace with your input file name
output_file = 'Draco_xwell_predict_clean.csv'  # Replace with your output file name

# Determine the file format
file_extension = os.path.splitext(input_file)[1]

# Load the data into a pandas DataFrame based on the file format
if file_extension == '.xlsx':
    df = pd.read_excel(input_file)
elif file_extension == '.csv':
    df = pd.read_csv(input_file)
else:
    raise ValueError("Unsupported file format. Only CSV and XLSX are supported.")

# Apply the conditions and update the POR_N column
condition = ((df['DPHI_N'] < 0) | (df['NPHI_N'] < 0) | ((df['DPHI_N'] - df['NPHI_N']) > 0.14))
df.loc[condition, 'POR_N'] = 0

# Save the modified DataFrame back to the same format
if file_extension == '.xlsx':
    df.to_excel(output_file, index=False)
elif file_extension == '.csv':
    df.to_csv(output_file, index=False)

print("Processing complete. Data saved to", output_file)