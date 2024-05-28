from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from pictor.xomics.omix import Omix
from roma import finder, console

import os



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
data_dir = r'D:\data\BAMIA\CT'
feature_dir = os.path.join(data_dir, 'rad_features_pool')

# -----------------------------------------------------------------------------
# Read all omix and merge
# -----------------------------------------------------------------------------
omix_file_paths = finder.walk(feature_dir, pattern='*.omix')
omix_list = [Omix.load(omix_file_path) for omix_file_path in omix_file_paths]

omix = Omix.sum(omix_list, data_name='BAMIA-Rad')
omix.show_in_explorer()
