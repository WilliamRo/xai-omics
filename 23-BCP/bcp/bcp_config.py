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

    slice_num = Flag.integer(3, '', is_key=None)
    xy_size = Flag.integer(192, '', is_key=None)

    link_indices_str = Flag.string('a', 'U-Net link indices', is_key=None)

    filter = Flag.integer(8, 'filter size of Unet', is_key=None)
    kernel_size = Flag.integer(5, 'kernel size of Unet', is_key=None)
    depth = Flag.integer(3, 'depth of Unet', is_key=None)
    width = Flag.integer(2, 'width of Unet', is_key=None)
    activation = Flag.string('relu', 'activation of Unet', is_key=None)

    @property
    def link_indices(self):
        if self.link_indices_str in ('a', 'all', '-', ''):
            return self.link_indices_str
        return [int(s) for s in self.link_indices_str.split(',')]


# New hub class inherited from SmartTrainerHub must be registered
BCPConfig.register()
