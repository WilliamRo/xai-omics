import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly0
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import Classifier

from mic.mic_config import MICConfig as Hub

import mic_du as du


# -----------------------------------------------------------------------------
# Initialize config and set datas/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, '22-MIC/data')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.8

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.use_mask = False
th.window = [-1200, 0]
th.crop_size = [32, 64, 64]
th.input_shape = th.crop_size + [2] if th.use_mask else th.crop_size + [1]
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = False
th.patience = 5

th.print_cycle = 2
th.updates_per_round = 50
th.validation_per_round = 1

th.export_tensors_upon_validation = True

th.validate_train_set = True
th.validate_test_set = True

th.evaluate_train_set = True
th.evaluate_test_set = True



def activate():
  if 'deactivate' in th.developer_code: return

  # Load datas

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate. Note that, although both validation and evaluation use
  #  data_set.data_for_validation, evaluate_denoiser is called by data_set
  #  itself.

  train_set, val_set, test_set = du.load_data()
  val_set = val_set.dataset_for_eval
  test_set = test_set.dataset_for_eval
  if th.train:
    model.train(training_set=train_set, validation_set=val_set,
                test_set=test_set, trainer_hub=th)
  else:
    # data_set.evaluate_denoiser(model)
    # test_set.test_model(model)
    # val_set.test_model(model)
    # train_set.test_model(model)

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 'AUC'
    datasets = (train_set, val_set, test_set)
    for ds in datasets:
      ds = val_set
      results = model.predict(ds, th.eval_batch_size)
      fpr, tpr, thresholds = roc_curve(ds.targets[:, 1], results[:, 1])
      roc_auc = auc(fpr, tpr)
      plt.figure()
      plt.plot(fpr, tpr, color='darkorange', lw=2,
               label='ROC curve (AUC = {:.2f})'.format(roc_auc))
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'Receiver Operating Characteristic Curve -- {ds.name}')
      plt.legend(loc='lower right')
      plt.show()

  # model.agent.load()
  # for ds in (train_set, val_set, test_set):
  #   model.evaluate_pro(
  #     ds, batch_size=1, show_class_detail=True, export_false=True)



  # End
  model.shutdown()
  console.end()


if __name__ == "__main__":
  print(ROOT)