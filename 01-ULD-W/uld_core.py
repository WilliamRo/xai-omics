import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
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
from tframe import Predictor
from uld.uld_config import ULDConfig as Hub

import uld_du as du
import uld_tu as tu



# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, 'data')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = True
th.gpu_memory_fraction = 0.7

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.input_shape = [None, None, None, 1]
# th.input_shape = [640, 512, 512, 1]
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5

th.print_cycle = 2
th.updates_per_round = 50
th.validation_per_round = 2
th.val_batch_size = 1

th.export_tensors_upon_validation = True


def activate():
  if 'deactivate' in th.developer_code: return

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Predictor)

  # Load data
  train_set, val_set, test_set = du.load_data()

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate. Note that, although both validation and evaluation use
  #  data_set.data_for_validation, evaluate_denoiser is called by data_set
  #  itself.
  # th.additional_datasets_for_validation.append(some_data_set)
  if th.train:
    model.train(training_set=train_set, validation_set=val_set,
                test_set=test_set, trainer_hub=th, probe=tu.probe)
  else:
    test_set.evaluate_model(model)

  # End
  model.shutdown()
  console.end()


if __name__ == "__main__":
  print(ROOT)