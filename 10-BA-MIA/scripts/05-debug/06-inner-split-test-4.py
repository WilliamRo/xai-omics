from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline, FitPackage
from pictor.xomics.ml.dr.dr_engine import DREngine
from pictor.xomics.ml import get_model_class
from pictor.xomics.ml.ml_engine import MLEngine
from roma import console

import warnings, os
import numpy as np

warnings.simplefilter('ignore')
os.environ["PYTHONWARNINGS"] = "ignore"



def fit_k_fold(self, omix: Omix) -> FitPackage:
  prompt = '[K_FOLD_FIT] >>'
  # (0) Get settings
  n_splits = 5
  shuffle = True

  nested_prefix = 'Nested '
  console.section(f'{nested_prefix}K-fold fitting using {self}')

  #(1) Fit data in k-fold manner
  models, prob_list, pred_list, fold_pkgs = [], [], [], []
  k_fold_data, om_whole = omix.get_k_folds(
    k=n_splits, shuffle=shuffle, return_whole=True)

  for i, (om_train, om_test) in enumerate(k_fold_data):
    # Tune parameters on om_train if necessary
    hp = self.tune_hyperparameters(om_train, verbose=0)
    # console.show_status(f'Best hyperparameters: {hp}', prompt='TUNE')

    # Fit the model
    model = self.fit(om_train, hp=hp)

    # Pack the results
    prob = model.predict_proba(om_test.features)
    pred = model.predict(om_test.features)

    prob_list.append(prob)
    pred_list.append(pred)
    models.append(model)

    # Pack sub-package
    sub_pkg = FitPackage.pack(pred, prob, om_test, hp=hp)
    fold_pkgs.append(sub_pkg)

    # Print results
    # for x in om_test.features
    console.show_info(f'Fold-{i+1}: train # {om_train.n_samples}, test # {om_test.n_samples}')
    console.supplement(f'AUC = {sub_pkg.ROC.auc:.3f}', level=2)

  # probabilities.shape = (n_samples, n_classes)
  probabilities = np.concatenate(prob_list, axis=0)
  predictions = np.concatenate(pred_list)

  console.show_status('Fitting completed.', prompt=prompt)

  # (3) Analyze results if required
  package = FitPackage.pack(predictions, probabilities, om_whole, models, hp,
                            sub_packages=fold_pkgs)

  # (-1) Return the fitted models and probabilities
  return package

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
console.section('Select features for omix_1')
omix_1.report(), print()
omix_1_reduced = omix_1.select_features('lasso', lasso_repeats=5, verbose=1,
                                        save_model=True)
lasso: DREngine = omix_1_reduced.dimension_reducer

# TODO: Checkpoint
if 0: omix_1_reduced.show_in_explorer(), exit()

# 2.2 Fit k-fold
console.section('Nested 5-fold fitting using LR')
ModelClass = get_model_class('lr')
model: MLEngine = ModelClass()
pkg = fit_k_fold(model, omix_1_reduced)

console.show_info('AUC for each fold:')
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


