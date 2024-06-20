from roma import Arguments
from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag


class RLDConfig(SmartTrainerHub):

  windows_size = Flag.integer(128, 'Window size during training', is_key=None)
  eval_windows_size = Flag.integer(128, 'Window size for validation', is_key=None)

  buffer_size = Flag.integer(6, 'Number of subject groups loaded per round',
                             is_key=None)

  use_suv = Flag.boolean(True, 'Whether transform the data to SUV',
                         is_key=None)

  use_sigmoid = Flag.boolean(False, "whether use sigmoid", is_key=None)
  data_shape = Flag.list(None, "the dataset shape for model")
  opt_str = Flag.string('adam', "the string of optimizer", is_key=None)

  clip_off = Flag.boolean(False, '...', is_key=None)

  noCT = Flag.boolean(False, 'if use ct in input', is_key=None)
  data_set = Flag.list(None, 'select which data to train')

  show_weight_map = Flag.boolean(False, '...', is_key=None)
  output_conv = Flag.boolean(True, 'use conv at end of net', is_key=None)
  norm_method = Flag.string(None, 'normalization methods', is_key=None)

  use_res = Flag.boolean(False, 'use residual link', is_key=None)
  gen_test_nii = Flag.boolean(False, '...', is_key=None)
  process_param = Flag.whatever({}, 'data relevant parameters', is_key=None)

  internal_loss = Flag.boolean(False, 'use internal loss', is_key=None)

  statistics = Flag.boolean(False, 'use statistics', is_key=None)

  use_seg = Flag.whatever(None, 'use segmentation', is_key=None)
  extra_data = Flag.whatever([], 'extra data for feature', is_key=None)

  gan = Flag.boolean(False, 'use the gan model', is_key=None)
  ddpm = Flag.boolean(False, 'use the ddpm model', is_key=None)
  time_step = Flag.integer(1000, 'time step for DDPM', is_key=None)
  time_dim = Flag.integer(128, 'time dimension for DDPM', is_key=None)

  dimension = Flag.integer(3, 'dimension of the data', is_key=None)
  gen_gaussian = Flag.integer(0, 'gaussian sigma', is_key=None)
  gen_dcm = Flag.boolean(False, 'generate dicom', is_key=None)
  gen_mask = Flag.boolean(False, 'generate mask', is_key=None)

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
