from tframe.core.quantity import Quantity
from tframe import console
from tframe import mu

import numpy as np



def get_initial_model():
  from uld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from uld_core import th

  assert isinstance(model, mu.Predictor)
  model.add(mu.HyperConv3D(filters=1, kernel_size=1, activation='sigmoid'))

  # Build model
  # model.build(loss=th.loss_string, metric=['loss'])
  model.build(loss=th.loss_string, metric=[get_ssim_3D(), 'loss'])
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_ssim_3D():
  from tframe import tf

  def ssim(truth, output):
    # [bs, num_slides, 440, 440, 1]
    shape = [-1, 440, 440, 1]
    truth, output = [tf.reshape(x, shape) for x in (truth, output)]

    return tf.image.ssim(truth, output, max_val=1.0)

  return Quantity(ssim, tf.reduce_mean, name='SSIM', lower_is_better=False)
