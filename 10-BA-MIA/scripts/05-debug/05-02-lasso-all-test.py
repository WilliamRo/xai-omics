from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from pictor.xomics.ml.dr.dr_engine import DREngine
from pictor.xomics.ml import get_model_class
from pictor.xomics.ml.ml_engine import MLEngine
from roma import console

import warnings, os

warnings.simplefilter('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"



# ------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------
data_path = r'D:/data/BAMIA/BAMIA-All-851.omix'

omix = Omix.load(data_path)
omix_1, omix_2 = omix.split(1, 1)

assert isinstance(omix_1, Omix) and isinstance(omix_2, Omix)

# TODO: Checkpoint
if 0: omix_1.show_in_explorer(), exit()
# ------------------------------------------------------------------------
# 2. Construct pipeline on omix_1
# ------------------------------------------------------------------------

# 2.1 Feature selection useing LASSO
console.section('Select features')
omix_1.report(), print()

use_global_reducer = 0
sf_method = ['lasso', 'pval'][0]
kwargs = {'k': 30}
if use_global_reducer:
  omix_all_reduced = omix.select_features(
    sf_method, lasso_repeats=5, verbose=1, save_model=True, **kwargs)
  lasso: DREngine = omix_all_reduced.dimension_reducer
  omix_1_reduced = lasso.reduce_dimension(omix_1)
else:
  omix_1_reduced = omix_1.select_features(
    sf_method, lasso_repeats=5, verbose=1, save_model=True, **kwargs)
  lasso: DREngine = omix_1_reduced.dimension_reducer


# TODO: Checkpoint
if 0: omix_1_reduced.show_in_explorer(), exit()

# 2.2 Fit k-fold
console.section('Nested 5-fold fitting using LR')
ModelClass = get_model_class('lr')
model: MLEngine = ModelClass()
pkg = model.fit_k_fold(omix_1_reduced, save_models=True, nested= True)

console.show_info('AUC for each fold:')
auc_str = ', '.join([f'{spkg.ROC.auc:.3f}' for spkg in pkg.sub_packages])
console.supplement(auc_str, level=2)
console.show_info('Overall confusion matrix and classification report:')
pkg.report()
# ------------------------------------------------------------------------
# 3. Evaluate pipeline on omix_2
# ------------------------------------------------------------------------
console.section('Evaluation on omix_2')
omix_2.report(), print()
omix_2_reduced = lasso.reduce_dimension(omix_2)
pkg_2 = pkg.evaluate(omix_2_reduced)
console.show_info('Overall confusion matrix and classification report:')
pkg_2.report()

auc_of = pkg.ROC.auc - pkg_2.ROC.auc
console.show_info(f'AUC overfit = {auc_of: .3f}')


