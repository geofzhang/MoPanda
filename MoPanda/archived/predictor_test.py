from EDA.MoPanda.MoPanda.modules.inwellpredictor import InWellPredictor
from EDA.MoPanda.MoPanda.modules.las_io import LasIO

las_file_path: str = '../data/las/Denova1_modified.LAS'

# Load LAS file
log = LasIO(las_file_path)
InWellPredictor(log)
