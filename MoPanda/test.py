from Log import Log
from graphs import LogViewer
from electrofacies_GUI import electrofacies
from utils import ColorCoding as cc
import os

las_file_path: str = './data/las/Denova1_modified.las'
tops_file_path = './data/log_info/tops.csv'
xml_template = 'permeability'
lithology_color_coding = './data/color_code/lithology_color_code.xml'
excel_output = './output/Denova1_test.xlsx'
start_depth = 1000
end_depth = 5000
masking = {
    'status': False,
    'mode': 'white',
    'facies_to_drop': ['Silty Shale', 'Shaly Sandstone', 'Shale', 'Black Shale', 'Halite', 'Anhydrite', 'Gypsum',
                       'Anomaly', 'Else'],
    'curves_to_mask': ['SALINITY_N', 'RWA_N'],
}

# Load the color-label-lithology relationship
color = cc().litho_color(lithology_color_coding)

# Load LAS file
log = Log(las_file_path)

# # Display well head information
# print(log.well)
#
# # Display drilling parameters
# print(log.params)

# # Export converted data (raw) to either .csv or .xlsx
# log.export_csv()
# log.export_excel()

# Initial log QC and selection
# Set the change of curves and save it back to LASFile.
"""
the DataFrame converted by LASFile uses the DEPT as index
If you prefer the DEPT curve not to be set as the index
Use the following command to reset index:

df2 = log.df().reset_index()
"""
df = log.log_qc(start_depth, end_depth)

# Auto aliasing log names
log.aliasing()
# print(log.curves)

# Load formation tops
log.load_tops(csv_path=tops_file_path, depth_type='MD', source='CA')

# # View and modify logs
# viewer = LogViewer(log, top=3950, height=1000)
# viewer.show()

# # Calculate formation fluid property parameters
# log.load_fluid_properties()
# log.formation_fluid_properties(formations=[], parameter='default')
#
# # Calculate multimineral model
# log.load_multilateral_parameters()
# log.formation_multimineral_model(formations=[], parameter='default')

# Electrofacies
logs = [log]  # List of Log objects
formations = ['SKULL_CREEK_SH', 'LAKOTA_UPPER', 'LAKOTA_LOWER', 'MORRISON',
              'DAYCREEK', 'FLOWERPOT_SH', 'LYONS', 'SUMNER_SATANKA',
              'STONE_CORRAL']  # List of formation names (optional)
curves = []
# curves = ['CGR_N', 'SP_N', 'NPHI_N', 'DPHI_N', 'PE_N', 'SGR_N',
#           'RESSHAL_N', 'RESDEEP_N', 'DTS_N', 'DTC_N', 'TCMR', 'T2LM', 'RHOMAA_N', 'UMAA_N',
#           'RWA_N']  # List of curve names (optional)
log_scale = ['RESSHAL_N', 'RESDEEP_N']  # List of curve names to preprocess on a log scale (optional)
n_components = 0.85  # Number of principal components to keep (optional)
curve_names = []  # List of names for output electrofacies curves (optional)
clustering_methods = ['kmeans', 'dbscan', 'affinity', 'agglom',
                      'fuzzy']  # List of clustering methods to be used (optional)
cluster_range = (2, 10)
clustering_params = {
    'kmeans': {'n_clusters': 12, 'n_init': 3},  # "n_clusters" is optional if auto optimization is wanted
    'dbscan': {'eps': 0.8, 'min_samples': 8},
    'affinity': {'random_state': 20, 'affinity': 'euclidean'},
    'optics': {'min_samples': 20, 'max_eps': 0.5, 'xi': 0.05},
    'agglom': {'n_clusters': 12},
    'fuzzy': {'n_clusters': 9}  # "n_clusters" is optional if auto optimization is wanted
}

if xml_template == 'electrofacies':
    output_template, _ = electrofacies(logs, formations, curves, log_scale=log_scale,
                                       n_components=n_components, curve_names=curve_names,
                                       clustering_methods=clustering_methods,
                                       clustering_params=clustering_params,
                                       template=xml_template,
                                       lithology_color_coding=lithology_color_coding,
                                       masking=masking)
else:
    electrofacies(logs, formations, curves, log_scale=log_scale,
                  n_components=n_components, curve_names=curve_names,
                  clustering_methods=['kmeans', 'agglom'],
                  clustering_params=clustering_params,
                  template=xml_template,
                  lithology_color_coding=lithology_color_coding,
                  masking=masking)
print(log.curves)

# View and modify logs
viewer = LogViewer(log, template_defaults=xml_template, top=4500, height=500, masking=masking)
viewer.show()

# Export converted data (raw) to either .csv or .xlsx
log.export_excel(excel_output)
