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
# 2.1 Feature selection
console.section('Select features for omix_1')
omix_1.report(), print()
omix_1_reduced = omix_1.select_features(
  'pval', k=20, verbose=1, save_model=True)
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
omix_2_reduced = lasso.reduce_dimension(omix_2)
omix_2_reduced.report(), print()
pkg_2 = pkg.evaluate(omix_2_reduced)
console.show_info('Overall confusion matrix and classification report:')
pkg_2.report()


