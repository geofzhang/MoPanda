from Log import Log
from graphs import LogViewer
from electrofacies_GUI import electrofacies

las_file_path: str = './data/Denova1_modified.las'
tops_file_path = './data/tops.csv'
start_depth = 1000
end_depth = 5000

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
# print(log.tops)

# # View and modify logs
# viewer = LogViewer(log, top=3950, height=1000)
# viewer.show()



# Electrofacies
logs = [log]  # List of Log objects
formations = []  # List of formation names (optional)
curves = []  # List of curve names (optional)
log_scale = ['RESDEEP_N']  # List of curve names to preprocess on a log scale (optional)
n_components = 0.85  # Number of principal components to keep (optional)
curve_names = []  # List of names for output electrofacies curves (optional)
clustering_methods = []  # List of clustering methods to be used (optional)
cluster_range = (2, 10)
clustering_params = {
    'kmeans': {'n_init': 'auto'},  # 'n_clusters' is optional if auto optimization is wanted
    'dbscan': {'eps': 0.5, 'min_samples': 10},
    'affinity': {'affinity': 'euclidean'},
    'optics': {'min_samples': 10},
    'fuzzy': {}  # 'n_clusters' is optional if auto optimization is wanted
}

# electrofacies(logs, formations, curves, log_scale=log_scale,
#               n_components=n_components, curve_names=curve_names,
#               clustering_methods=clustering_methods,
#               clustering_params=clustering_params)
print(log.curves)
