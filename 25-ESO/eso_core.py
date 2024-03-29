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
from tframe import Predictor

from eso.eso_config import ESOConfig as Hub

import eso_du as du
import eso_tu as tu


# -----------------------------------------------------------------------------
# Initialize config and set datas/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = os.path.join(ROOT, '25-ESO/data')

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.8

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
th.ratio_of_dataset = '80:10:18'
th.window = [-300, 400]
th.crop_size = [128, 512, 512]
# th.input_shape = [None, None, None, 3]
th.input_shape = [128, 256, 256, 3]

th.random_flip = True
th.random_rotation = True
th.random_noise = True
# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 5

th.print_cycle = 2
th.updates_per_round = 20
th.validation_per_round = 1

th.export_tensors_upon_validation = True

th.validate_train_set = True
th.validate_test_set = True

th.evaluate_train_set = True
th.evaluate_test_set = True



def activate():
  if 'deactivate' in th.developer_code: return

  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Predictor)

  print(f'th.archi_string: {th.archi_string}\n')
  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')

    # TODO
    structure = model.structure_detail[0]
    element = structure.split('\n')
    parameter_total = 0
    pattern = r' \d+x\d+x\d+x\d+ '
    import re
    from functools import reduce

    for i, e in enumerate(element):
      result = re.search(pattern, e)
      if result:
        size = [int(s) for s in result.group().split('x')]
        parameter = reduce(lambda x, y: x * y, size)
        parameter = parameter * 4.0 / 1024.0 / 1024.0
        element[i] = e + '\t' + f'{round(parameter, 2)} M'
        parameter_total = parameter_total + parameter

    element[-2] = element[-2] + '\t' + f'{round(parameter_total, 2)} M'

    structure_detail = '\n'.join(element)
    print(structure_detail)

    return

  # Load datas
  train_set, val_set, test_set = du.load_data()

  # Train or evaluate. Note that, although both validation and evaluation use
  #  data_set.data_for_validation, evaluate_denoiser is called by data_set
  #  itself.
  if th.train:
    model.train(training_set=train_set, validation_set=val_set,
                test_set=test_set, trainer_hub=th, probe=tu.probe,
                evaluate=tu.evaluate)
  else:
    from eso.eso_set import ESOSet
    assert isinstance(test_set, ESOSet)
    test_set.test_model(model)
    val_set.test_model(model)
    train_set.test_model(model)
    # test_set.get_pred_mask_as_nii(model)
    # val_set.get_pred_mask_as_nii(model)
    # train_set.get_pred_mask_as_nii(model)


  # End
  model.shutdown()
  console.end()


if __name__ == "__main__":
  print(ROOT)
