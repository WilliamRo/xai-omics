from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class ULDConfig(SmartTrainerHub):

  window_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_window_size = Flag.integer(128, 'Window size for validation', is_key=None)

  learn_delta = Flag.boolean(True, 'Whether to add shortcut at the end',
                             is_key=None)
  slice_size = Flag.integer(16, 'Slice size during training', is_key=None)

  @property
  def data_arg(self) -> Arguments:
    return Arguments.parse(self.data_config)


# New hub class inherited from SmartTrainerHub must be registered
ULDConfig.register()