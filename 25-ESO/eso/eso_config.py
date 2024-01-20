from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class ESOConfig(SmartTrainerHub):
  is_multiple = Flag.boolean(True, "1", is_key=None)

  # Data
  ratio_of_dataset = Flag.string(
    '7:2:1', 'the ratio of dataset', is_key=None)
  window = Flag.list([-300, 400], 'the window for image')
  crop_size = Flag.list([64, 256, 256], 'the size of image crop')
  random_flip = Flag.boolean(
    False, 'whether to flip the image', is_key=None)
  random_rotation = Flag.boolean(
    False, 'whether to rotate the image', is_key=None)
  random_noise = Flag.boolean(
    False, 'whether to add noise to image', is_key=None)
  random_crop = Flag.boolean(
    False, 'whether to crop image randomly', is_key=None)

  # Model
  filter = Flag.integer(8, 'filter size of Unet', is_key=None)
  kernel_size = Flag.integer(5, 'kernel size of Unet', is_key=None)
  depth = Flag.integer(3, 'depth of Unet', is_key=None)
  width = Flag.integer(2, 'width of Unet', is_key=None)
  activation = Flag.string('relu', 'activation of Unet', is_key=None)

  # Sundries
  cross_validation = Flag.boolean(
    False, 'whether to use cross validation', is_key=None)
  k_fold = Flag.integer(5, 'Fold number', is_key=None)
  val_fold_index = Flag.integer(0, 'Number of validation fold', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
ESOConfig.register()
