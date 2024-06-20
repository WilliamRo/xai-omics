from pictor.xomics import Omix
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'D:/data/BAMIA/'
file_name = [
  'BAMIA-All-851.omix',
  'BAMIA-Origin-107.omix',
][1]

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
omix = Omix.load(os.path.join(data_dir, file_name))

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
pi = Pipeline(omix, ignore_warnings=1, save_models=0)

M = 2
k = 5
pi.create_sub_space('pca', n_components=k, repeats=M, show_progress=1)
# pi.create_sub_space('sig', n_components=k, repeats=M, show_progress=1)

N = 3
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)

pi.report()
