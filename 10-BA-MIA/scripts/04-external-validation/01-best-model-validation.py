from pictor.xomics import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from pictor.xomics.ml.dr.dr_engine import DREngine
from pictor.xomics.ml.ml_engine import FitPackage

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

# (0) Report and show matrix if necessary
pi.report()

# (1) Evaluate the best model
ranking = pi.pipeline_ranking
print('AUC:', [r[0] for r in ranking])

best_dr: DREngine = ranking[0][1]
best_pkg: FitPackage = ranking[0][2]

ext_omix_reduced = best_dr.reduce_dimension(ext_omix)
pkg = best_pkg.evaluate(ext_omix_reduced)

pkg.report(show_signature=1, omix=ext_omix_reduced)
