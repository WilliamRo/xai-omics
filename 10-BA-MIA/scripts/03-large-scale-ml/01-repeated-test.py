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
][0]

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
omix = Omix.load(os.path.join(data_dir, file_name))

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
pi = Pipeline(omix, ignore_warnings=1, save_models=0)

M = 10
pi.create_sub_space('lasso', repeats=M, show_progress=1)
pi.create_sub_space('pca', n_components=10, repeats=M, show_progress=1)
pi.create_sub_space('pca', n_components=20, repeats=M, show_progress=1)
pi.create_sub_space('mrmr', k=10, repeats=M, show_progress=1)
pi.create_sub_space('mrmr', k=20, repeats=M, show_progress=1)

N = 10
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)
pi.fit_traverse_spaces('svm', repeats=N, show_progress=1)
pi.fit_traverse_spaces('dt', repeats=N, show_progress=1)
pi.fit_traverse_spaces('rf', repeats=N, show_progress=1)
pi.fit_traverse_spaces('xgb', repeats=N, show_progress=1)

pi.report()

omix.save(os.path.join(data_dir, '20240529-All.omix'), verbose=True)
