from EDA.MoPanda.MoPanda.modules.data_analysis import InWellPredictor
from EDA.MoPanda.MoPanda.modules.las_io import LasIO
import os
las_file_path: str = './data/las/Denova1_modified.las'

# Load LAS file
log = LasIO(las_file_path)
InWellPredictor(log)

