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

# (0) Report and show matrix if necessary
pi.report()
if 0: pi.plot_matrix()

# (1) Feature selection
# TODO: select best pipeline's feature selection method
sub_space_0: Omix = pi.sub_spaces[0]
lasso_0 = sub_space_0['sf_method']

ext_omix_reduced = ext_omix.get_sub_space(
  lasso_0.selected_indices, start_from_1=False)

# (1.1) Compare selected features
if 0:
  ext_omix_reduced.show_in_explorer()
  sub_space_0.show_in_explorer()
  exit(0)

# (2) Run model


# print(lasso_0.selected_indices)
# omix_0: Omix = lasso_0.select_features(ext_omix)
# omix_0.show_in_explorer()

print()

