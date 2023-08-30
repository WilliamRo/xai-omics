import numpy as np

from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class ULDConfig(SmartTrainerHub):

  window_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_window_size = Flag.integer(128, 'Window size for validation', is_key=None)

  learn_delta = Flag.boolean(False, 'Whether to add shortcut at the end',
                             is_key=None)
  slice_size = Flag.integer(16, 'Slice size during training', is_key=None)
  buffer_size = Flag.integer(6, 'Number of subject groups loaded per round',
                                is_key=None)

  color_map = Flag.string(None, 'use color map to transform the data to rgb',
                           is_key=None)
  use_suv = Flag.boolean(False, 'Whether transform the data to SUV',
                         is_key=None)
  use_tanh = Flag.float(0.0, 'use tanh(kx) to preprocess the data',
                        is_key=None)
  max_clip = Flag.float(None, "clip the data max value", is_key=None)

  use_sigmoid = Flag.boolean(False, "whether use sigmoid", is_key=None)
  train_self = Flag.boolean(False, "let feature and target are same", is_key=None)

  norm_by_feature = Flag.boolean(True, 'Whether use feature set to normalize '
                                        'the target', is_key=None)
  data_shape = Flag.list(None, "the dataset shape for model")
  rand_batch = Flag.boolean(True, "whether generate batch with true random",
                            is_key=None)

  @property
  def data_arg(self) -> Arguments:
    return Arguments.parse(self.data_config)

  @property
  def exp_name(self):
    return Arguments.parse(self.data_config).func_name

  @property
  def data_kwargs(self) -> dict:
    return Arguments.parse(self.data_config).arg_dict



# New hub class inherited from SmartTrainerHub must be registered
ULDConfig.register()