from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class BCPConfig(SmartTrainerHub):

    ratio_of_dataset = Flag.string(
        '7:2:1', 'the ratio of dataset', is_key=None)
    crop_size = Flag.list([32, 64, 64], 'the size of image crop')
    window = Flag.list([-1200, 0], 'the window for image')

    random_flip = Flag.boolean(
        False, 'whether to flip the image', is_key=None)
    random_rotation = Flag.boolean(
        False, 'whether to rotate the image', is_key=None)
    random_noise = Flag.boolean(
        False, 'whether to add noise to image', is_key=None)


# New hub class inherited from SmartTrainerHub must be registered
BCPConfig.register()
