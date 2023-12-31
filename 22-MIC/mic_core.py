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
  train_set, val_set, test_set = du.load_data()
  val_set = val_set.dataset_for_eval
  test_set = test_set.dataset_for_eval

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
  if th.train:
    model.train(training_set=train_set, validation_set=val_set,
                test_set=test_set, trainer_hub=th)
  else:
    # data_set.evaluate_denoiser(model)
    # test_set.test_model(model)
    val_set.test_model(model)

  # model.agent.load()
  # for ds in (train_set, val_set, test_set):
  #   model.evaluate_pro(
  #     ds, batch_size=1, show_class_detail=True, export_false=True)



  # End
  model.shutdown()
  console.end()


if __name__ == "__main__":
  print(ROOT)