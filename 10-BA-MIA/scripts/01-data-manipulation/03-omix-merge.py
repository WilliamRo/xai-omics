from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from pictor.xomics.omix import Omix
from roma import finder, console

import os



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
data_dir = r'D:/data/BAMIA/'
rad_fn = 'BAMIA-All-851.omix'
path_fn = 'patho_0526_fcn(1)_2_Sc_177.omix'

# -----------------------------------------------------------------------------
# Read all omix and merge
# -----------------------------------------------------------------------------
rad_omix = Omix.load(os.path.join(data_dir, rad_fn))
path_omix = Omix.load(os.path.join(data_dir, path_fn))
path_omix = path_omix.duplicate(target_labels=['BA', 'MIA'],
                                data_name='BAMIA-Patho')

# print(path_omix.sample_labels)

omix = rad_omix.intersect_merge(path_omix)
# omix.show_in_explorer()

