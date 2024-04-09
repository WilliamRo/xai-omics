from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class PROConfig(SmartTrainerHub):
  ratio_of_dataset = Flag.string(
    '7:1:2', 'the ratio of dataset', is_key=None)

  # Data
  label_type = Flag.string('trg_01_23', '', is_key=None)
  k_fold = Flag.integer(10, '', is_key=None)
  val_fold_index = Flag.integer(0, '', is_key=None)
  cross_validation = Flag.boolean(False, '', is_key=None)

  # Model
  dim = Flag.integer(3, '', is_key=None)
  model_name = Flag.string('', '', is_key=None)

# New hub class inherited from SmartTrainerHub must be registered
PROConfig.register()
