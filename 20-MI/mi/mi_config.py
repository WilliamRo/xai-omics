from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class MIConfig(SmartTrainerHub):

  is_multiple = Flag.boolean(True, "1", is_key=None)
  ratio_of_dataset = Flag.string(
    '7:2:1', 'the ratio of dataset', is_key=None)
  read_from_hd = Flag.boolean(
    False, 'whether to read data from the hard disk', is_key=None)
  use_pet = Flag.boolean(True, 'whether to use pet data', is_key=None)
  window = Flag.list([-300, 400], 'the window for image')
  crop_size = Flag.list([64, 256, 256], 'the size of image crop')

  random_flip = Flag.boolean(
    False, 'whether to flip the image', is_key=None)
  random_rotation = Flag.boolean(
    False, 'whether to rotate the image', is_key=None)
  random_noise = Flag.boolean(
    False, 'whether to add noise to image', is_key=None)

  if_predict = Flag.boolean(False, '', is_key=None)



# New hub class inherited from SmartTrainerHub must be registered
MIConfig.register()
