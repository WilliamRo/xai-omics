from tframe import console
from tframe import mu
from tframe import tf
from tframe.core import init_with_graph
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

import numpy as np



def get_initial_model():
  from bcp_core import th

  # model = mu.Predictor(th.mark)
  model = mu.Classifier(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from bcp_core import th
  # assert isinstance(model, mu.Predictor)
  #
  # model.add(mu.HyperConv3D(filters=1, kernel_size=1))
  #
  # # Build model
  #
  # model.build(loss='rmse', metric='rmse')
  # return model
  assert isinstance(model, mu.Classifier)

  model.add(GlobalAveragePooling3D())
  model.add(mu.Dense(4, use_bias=False, activation='softmax'))

  # Build model

  model.build(batch_metric='accuracy', loss='cross_entropy')
  return model


def get_unet_3d(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_fcn_3d(arc_string='2-4-2-lrelu'):
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


def get_cnn_3d(arc_string='64-p-32'):
  model = get_initial_model()
  kernel_size = 5
  activation = 'relu'
  use_batchnorm = False

  for i, c in enumerate(arc_string.split('-')):
    if c == 'p':
      model.add(mu.MaxPool3D(pool_size=2, strides=2))
      continue

    c = int(c)
    model.add(mu.HyperConv3D(
      filters=c, kernel_size=kernel_size, use_bias=False,
      activation=activation, use_batchnorm=use_batchnorm and i > 0))

  return finalize(model)


class GlobalAveragePooling3D(Layer):

  full_name = 'globalavgpool3d'
  abbreviation = 'gap3d'

  @init_with_graph
  def __init__(self, data_format='channels_last', flatten=True, **kwargs):
    self._data_format = data_format
    assert data_format == 'channels_last'
    self._flatten = flatten
    self._kwargs = kwargs

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    shape = input_.shape.as_list()
    assert len(shape) == 5
    output = tf.layers.average_pooling3d(
      input_, pool_size=shape[1:4], strides=(1, 1, 1),
      data_format=self._data_format)
    output = tf.reshape(output, shape=[-1, output.shape.as_list()[-1]])
    return output


if __name__ == '__main__':
  # ++ Blue box for model
  from bcp_core import th

