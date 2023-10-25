from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class RLDConfig(SmartTrainerHub):

  window_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_window_size = Flag.integer(128, 'Window size for validation', is_key=None)

  slice_size = Flag.integer(16, 'Slice size during training', is_key=None)
  buffer_size = Flag.integer(6, 'Number of subject groups loaded per round',
                             is_key=None)

  use_suv = Flag.boolean(True, 'Whether transform the data to SUV',
                         is_key=None)

  use_sigmoid = Flag.boolean(False, "whether use sigmoid", is_key=None)
  data_shape = Flag.list(None, "the dataset shape for model")
  opt_str = Flag.string('adam', "the string of optimizer", is_key=None)

  data_clip = Flag.float(None, "clip the data value", is_key=None)
  clip_off = Flag.boolean(False, '...', is_key=None)

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
RLDConfig.register()
