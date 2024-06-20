from pictor.xomics import Omix
from pictor.xomics.evaluation.pipeline import Pipeline

import os



# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = r'D:/data/BAMIA/'
package_path = '20240620-RP-LASSO-LR.omix'

external_omix_path = 'BAMIA-radiopathomics-72x883.omix'
# -----------------------------------------------------------------------------
# Load omices
# -----------------------------------------------------------------------------
pkg_omix: Omix = Omix.load(os.path.join(data_dir, package_path))
ext_omix: Omix = Omix.load(os.path.join(data_dir, external_omix_path))

# -----------------------------------------------------------------------------
# Load features and targets
# -----------------------------------------------------------------------------
pi = Pipeline(pkg_omix, ignore_warnings=1, save_models=1)

pi.report()
pi.plot_matrix()

