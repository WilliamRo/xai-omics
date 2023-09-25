from tframe import console
from tframe import mu

import numpy as np



def get_initial_model():
  from bcp_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from bcp_core import th
  assert isinstance(model, mu.Predictor)

  model.add(mu.HyperConv3D(filters=1, kernel_size=1))

  # Build model

  model.build(loss='rmse', metric='rmse')
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)

def model2(arc_string='2-4-2-lrelu'):
  model = get_initial_model()
  option = arc_string.split('-')
  filters, kernel_size, height = [int(op) for op in option[:3]]
  activation = option[3]
  stride = [2, 2, 2]

  for h in range(height):
    model.add(mu.HyperConv3D(
      filters=filters*(2**h), kernel_size=kernel_size,
      strides=stride, activation=activation))

  for h in range(height):
    model.add(mu.HyperDeconv3D(
      filters=filters*(2**(height-h-1)), kernel_size=kernel_size,
      strides=stride, activation=activation))

  return finalize(model)



if __name__ == '__main__':
  # ++ Blue box for model
  from bcp_core import th

