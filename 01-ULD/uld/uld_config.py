import numpy as np

from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class ULDConfig(SmartTrainerHub):

  window_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_window_size = Flag.integer(128, 'Window size for validation', is_key=None)

  learn_delta = Flag.boolean(True, 'Whether to add shortcut at the end',
                             is_key=None)
  slice_size = Flag.integer(16, 'Slice size during training', is_key=None)
  buffer_size = Flag.integer(6, 'Number of subject groups loaded per round',
                                is_key=None)

  use_color = Flag.boolean(False, 'Whether transform the data to rgb',
                           is_key=None)
  use_tanh = Flag.float(0.0, 'use tanh(kx) to preprocess the data',
                        is_key=None)
  use_clip = Flag.float(np.Inf, "clip the data value", is_key=None)

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