from pictor.xomics import Omix
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'D:/data/BAMIA/'
file_name = [
  'BAMIA-radiopathomics-72x883.omix',
][0]

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
omix = Omix.load(os.path.join(data_dir, file_name))

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
pi = Pipeline(omix, ignore_warnings=1, save_models=1)

M = 1
pi.create_sub_space('lasso', repeats=M, show_progress=1)

N = 5
pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)

pi.report()

omix.save(os.path.join(data_dir, '20240620-RP-LASSO-LR.omix'), verbose=True)
