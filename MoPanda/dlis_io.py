from dlisio import dlis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dlis_path = './data/dlis/DenovaSequestration_Denova1_TD__NEXT_CMRT-main.dlis'
f, *tail = dlis.load(dlis_path)

# Viewing the File’s Metadata

# Data Origin
origin, *origin_tail = f.origins

# Frame
for frame in f.frames:

    # Search through the channels for the index and obtain the units
    for channel in frame.channels:
        if channel.name == frame.index:
            depth_units = channel.units

    print(f'Frame Name: \t\t {frame.name}')
    print(f'Index Type: \t\t {frame.index_type}')
    print(f'Depth Interval: \t {frame.index_min} - {frame.index_max} {depth_units}')
    print(f'Depth Spacing: \t\t {frame.spacing} {depth_units}')
    print(f'Direction: \t\t {frame.direction}')
    print(f'Num of Channels: \t {len(frame.channels)}')
    print(f'Channel Names: \t\t {str(frame.channels)}')
    print('\n\n')


# Parameters within the DLIS File
def summary_dataframe(object, **kwargs):
    # Create an empty dataframe
    df = pd.DataFrame()

    # Iterate over each of the keyword arguments
    for i, (key, value) in enumerate(kwargs.items()):
        list_of_values = []

        # Iterate over each parameter and get the relevant key
        for item in object:
            # Account for any missing values.
            try:
                x = getattr(item, key)
                list_of_values.append(x)
            except:
                list_of_values.append('')
                continue

        # Add a new column to our data frame
        df[value] = list_of_values

    # Sort the dataframe by column 1 and return it
    return df.sort_values(df.columns[0])


param_df = summary_dataframe(f.parameters, name='Name', long_name='Long Name', values='Value')

# Hiding unwanted channels that may be in parameters.
# These two lines can be commented out to show them
# mask = param_df['Name'].isin(['R8', 'RR1', 'WITN', 'ENGI'])
# param_df = param_df[~mask]

print(param_df)


# Channels
channels = summary_dataframe(f.channels, name='Name', long_name='Long Name',
                             dimension='Dimension', units='Units', frame='Frame')
print(channels)

# tools
tools = summary_dataframe(f.tools, name='Name', description='Description')

