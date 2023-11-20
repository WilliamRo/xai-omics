from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class TESConfig(SmartTrainerHub):

  ratio_of_dataset = Flag.string(
    '7:2:1', 'the ratio of dataset', is_key=None)
  crop_size = Flag.list([64, 256, 256], 'the size of image crop')
  window = Flag.list([-300, 400], 'the window for image')

  random_flip = Flag.boolean(
    False, 'whether to flip the image', is_key=None)
  random_rotation = Flag.boolean(
    False, 'whether to rotate the image', is_key=None)
  random_noise = Flag.boolean(
    False, 'whether to add noise to image', is_key=None)
  random_crop = Flag.boolean(
    False, 'whether to crop image randomly', is_key=None)

  link_indices_str = Flag.string('a', 'U-Net link indices', is_key=None)

  slice_num = Flag.integer(3, '', is_key=None)
  dim = Flag.integer(3, '', is_key=None)
  xy_size = Flag.integer(256, '', is_key=None)

  @property
  def link_indices(self):
    if self.link_indices_str in ('a', 'all', '-', ''):
      return self.link_indices_str
    return [int(s) for s in self.link_indices_str.split(',')]


# New hub class inherited from SmartTrainerHub must be registered
TESConfig.register()
