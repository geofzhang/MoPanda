import os

import pandas as pd

from modules.cmr_permeability import GaussianDecomposition
from modules.las_io import LasIO

las_file_path: str = './data/las/Denova1_modified.las'
tops_file_path = './data/log_info/tops.csv'
t2_file_path = './data/cmr/Denova1_T2_DIST_3800FT.xlsx'
xml_template = 'permeability'
lithology_color_coding = './data/color_code/lithology_color_code.xml'
excel_output = './output/Denova1_perm.xlsx'
start_depth = 3800
end_depth = 5000
masking = {
    'status': False,
    'mode': 'white',
    'facies_to_drop': ['Silty Shale', 'Shaly Sandstone', 'Shale', 'Black Shale', 'Halite', 'Anhydrite', 'Gypsum',
                       'Anomaly', 'Else'],
    'curves_to_mask': ['SALINITY_N', 'RWA_N'],
}

# Load LAS file
log = LasIO(las_file_path)

# # Load the color-label-lithology relationship
# color = cc().litho_color(lithology_color_coding)
#
# df = log.log_qc(start_depth, end_depth)
#
# # Auto aliasing log names
# log.aliasing()
#
# # Load formation tops
# log.load_tops(csv_path=tops_file_path, depth_type='MD', source='CA')

# Calculating GaussianDecomposition
file_name = os.path.basename(t2_file_path)
df = pd.read_excel(t2_file_path)
num_components = 7

# Plot the Gaussian decomposition for a specific index
gd = GaussianDecomposition(df)
index = 1709
gd.decomposition_single(index=index, num_components=num_components, auto_search=False)

# # find way to name well, looking for well name#
# # or UWI or API #
#
# if len(log.well['WELL'].value) > 0:
#     well_name = log.well['WELL'].value
# elif len(str(log.well['UWI'].value)) > 0:
#     well_name = str(log.well['UWI'].value)
# elif len(log.well['API'].value) > 0:
#     well_name = str(log.well['API'].value)
# else:
#     well_name = 'UNKNOWN'
# well_name = well_name.replace('.', '')
#
# # View and modify logs
# viewer = LogViewer(log, template_defaults=xml_template, top=4500, height=500, lithology_color_coding=color,
#                    masking=masking)
# viewer.fig.set_size_inches(17, 11)
#
# # add well_name to title of LogViewer #
#
# viewer.fig.suptitle(well_name, fontweight='bold', fontsize=30)
#
# # add logo to top left corner #
#
# logo_im = plt.imread('./logo/ca_logo.png')
# logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
# logo_ax.imshow(logo_im)
# logo_ax.axis('off')
#
# viewer.show()
#
# # Export converted data (raw) to either .csv or .xlsx
# log.export_excel(excel_output)
