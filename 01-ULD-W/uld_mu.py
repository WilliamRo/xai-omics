from tframe import mu
from tframe import tf
from uld.operators.custom_layers import Tanh_k, Atanh_k, Clip
from uld.operators.custom_loss import get_ssim_3D, get_nrmse, get_psnr, \
  get_pw_rmse

from tframe.layers.hyper.conv import ConvBase
from archi.energy_norm import NormalizeEnergy
from archi.adaptive_merge import AdaptiveMerge

EPSILON = 0.001


def get_initial_model():
  from uld_core import th

  if th.classify:
    model = mu.Classifier(th.mark)
  else:
    model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.use_tanh != 0:
    model.add(Tanh_k(k=th.use_tanh))
  return model


def finalize(model):
  from uld_core import th
  from tframe import tf

  custom_loss = {
    'nrmse': get_nrmse(),
    'ssim': get_ssim_3D(),
    'psnr': get_psnr(),
    # 'pw_rmse': get_pw_rmse(EPSILON),
  }

  if th.normalize_energy:
    from archi.energy_norm import NormalizeEnergy
    model.add(NormalizeEnergy(model.input_, th.ne_gamma))

  # Build model
  metrics = list(custom_loss.values())
  model.build(loss=th.loss_string, metric=metrics + ['loss'])
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()
  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)



def gen_ecc_filter(self: ConvBase, filter_shape):
  from uld_core import th

  # filter_shape = [ks, ks, ks, in_c, out_c]
  k = tf.get_variable(
    'kernel', shape=filter_shape, dtype=th.dtype,
    initializer=self._weight_initializer)
  k = tf.abs(k)

  k_sum = tf.reduce_sum(k, axis=[0, 1, 2])
  eck = k / k_sum
  # eck = tf.nn.softmax(k, axis=[0, 1, 2])

  return eck



