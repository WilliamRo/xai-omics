from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class ULDConfig(SmartTrainerHub):

  window_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_window_size = Flag.integer(128, 'Window size for validation', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
ULDConfig.register()