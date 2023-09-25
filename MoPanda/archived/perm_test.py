import os

import pandas as pd
from matplotlib import pyplot as plt

from EDA.MoPanda.MoPanda.modules.cmr_permeability import GaussianDecomposition, perform_gaussian_decomposition
from EDA.MoPanda.MoPanda.modules.electrofacies import electrofacies
from EDA.MoPanda.MoPanda.modules.graphs import LogViewer
from EDA.MoPanda.MoPanda.modules.las_io import LasIO
from EDA.MoPanda.MoPanda.modules.utils import ColorCoding as cc

# Parameter Input
las_file_path = '../data/las/Denova1_modified.LAS'
tops_file_path = '../data/log_info/tops.csv'
t2_file_path = '../data/cmr/Denova1_T2_DIST_3800FT.xlsx'
xml_template = 'permeability'
lithology_color_coding = '../data/color_code/lithology_color_code.xml'
excel_output = '../output/Denova1_perm.xlsx'
las_output = '../output/Denova1_perm.las'
start_depth = 3800
end_depth = 5000
display_decomp = True
index = 0
masking = {
    'status': False,
    'mode': 'white',
    'facies_to_drop': ['Silty Shale', 'Shaly Sandstone', 'Shale', 'Black Shale', 'Halite', 'Anhydrite', 'Gypsum',
                       'Anomaly', 'Else'],
    'curves_to_mask': ['SALINITY_N', 'RWA_N'],
}

# Load LAS file
log = LasIO(las_file_path)

# Load the color-label-lithology relationship
color = cc().litho_color(lithology_color_coding)

df = log.log_qc(start_depth, end_depth)

# Auto aliasing log names
log.aliasing()

# Load formation tops
log.load_tops(csv_path=tops_file_path, depth_type='MD', source='CA')

# calculate permeability if permeability template is selected
if xml_template == 'permeability':
    # Calculating GaussianDecomposition
    file_name = os.path.basename(t2_file_path)
    df = pd.read_excel(t2_file_path)
    num_components = 7

    log = perform_gaussian_decomposition(log, df, file_name, num_components=num_components)
    if display_decomp:
        gd = GaussianDecomposition(df)
        # Plot the Gaussian decomposition for a specific index
        index = index
        gd.decomposition_single(index, num_components=7, auto_search=False)

# predictor = WellLogPredictor(log)

# Electrofacies
logs = [log]  # List of Log objects
formations = ['SKULL_CREEK_SH', 'LAKOTA_UPPER', 'LAKOTA_LOWER', 'MORRISON',
              'DAYCREEK', 'FLOWERPOT_SH', 'LYONS', 'SUMNER_SATANKA',
              'STONE_CORRAL']  # List of formation names (optional)
# curves = []
curves = ['CAL_N', 'RHOB_N', 'CGR_N', 'SP_N', 'NPHI_N', 'DPHI_N', 'PE_N', 'SGR_N',
          'RESSHAL_N', 'RESDEEP_N', 'DTS_N', 'DTC_N', 'TCMR', 'T2LM', 'RHOMAA_N', 'UMAA_N',
          'RWA_N']  # List of curve names (optional)
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
                  clustering_methods=['kmeans'],
                  clustering_params=clustering_params,
                  template=xml_template,
                  lithology_color_coding=lithology_color_coding,
                  masking=masking)
print(log.curves)

# find way to name well, looking for well name#
# or UWI or API #

if len(log.well['WELL'].value) > 0:
    well_name = log.well['WELL'].value
elif len(str(log.well['UWI'].value)) > 0:
    well_name = str(log.well['UWI'].value)
elif len(log.well['API'].value) > 0:
    well_name = str(log.well['API'].value)
else:
    well_name = 'UNKNOWN'
well_name = well_name.replace('.', '')

# View and modify logs
viewer = LogViewer(log, template_defaults=xml_template, top=4500, height=500, lithology_color_coding=color,
                   masking=masking)
viewer.fig.set_size_inches(17, 11)

# add well_name to title of LogViewer #

viewer.fig.suptitle(well_name, fontweight='bold', fontsize=30)

# add logo to top left corner #

logo_im = plt.imread('../logo/ca_logo.png')
logo_ax = viewer.fig.add_axes([0, 0.85, 0.2, 0.2])
logo_ax.imshow(logo_im)
logo_ax.axis('off')

viewer.show()

# Export converted data (raw) to either .csv or .xlsx
log.export_excel(excel_output)

log.write(las_output)
