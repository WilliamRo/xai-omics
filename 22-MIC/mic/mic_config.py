from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class MICConfig(SmartTrainerHub):

  ratio_of_dataset = Flag.string(
    '7:2:1', 'the ratio of dataset', is_key=None)
  crop_size = Flag.list([32, 64, 64], 'the size of image crop')
  window = Flag.list([-1200, 0], 'the window for image')

  random_flip = Flag.boolean(
    False, 'whether to flip the image', is_key=None)
  random_rotation = Flag.boolean(
    False, 'whether to rotate the image', is_key=None)
  random_translation = Flag.boolean(
    False, 'Option to toggle random translation in gen_batches',
    is_key=None)
  random_noise = Flag.boolean(
    False, 'whether to add noise to image', is_key=None)

  use_mask = Flag.boolean(
    False, 'whether to use mask as the second channel', is_key=None)

  cross_validation = Flag.boolean(
    False, 'whether to use cross validation', is_key=None)
  num_fold = Flag.integer(5, 'the number of the fold', is_key=None)
  # data_config = Flag.string('5:4', 'num_fold:valid_num', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
MICConfig.register()
