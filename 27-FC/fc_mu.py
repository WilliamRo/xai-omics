from tframe import console, context
from tframe import mu
from tframe import tf
from tframe.layers.layer import Layer
from tframe.core import init_with_graph
from tframe.layers.layer import single_input

import numpy as np



# Model
def get_initial_model():
  from fc_core import th

  model = mu.Classifier(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def get_container(flatten=False):
  from fc_core import th

  model = mu.Classifier(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model):
  from fc_core import th
  assert isinstance(model, mu.Classifier)

  # Build model
  model.add(mu.Dense(th.num_classes, use_bias=False))
  model.add(mu.Activation('softmax'))
  model.build(loss=th.loss_string, metric='f1')

  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_cnn_3d(archi_string):
  model = get_initial_model()

  from fc_core import th

  for i, c in enumerate(archi_string.split('-')):
    if c == 'p':
      model.add(mu.MaxPool3D(pool_size=2, strides=2))
      continue

    c = int(c)
    model.add(mu.HyperConv3D(
      filters=c, kernel_size=th.kernel_size, use_bias=False,
      activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))

  model.add(mu.GlobalAveragePooling3D())

  return finalize(model)


def get_cnn_2d(archi_string):
  model = get_initial_model()

  from fc_core import th

  for i, c in enumerate(archi_string.split('-')):
    if c == 'p':
      model.add(mu.MaxPool2D(pool_size=2, strides=2))
      continue

    c = int(c)
    model.add(mu.HyperConv2D(
      filters=c, kernel_size=th.kernel_size, use_bias=False,
      activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))

  model.add(mu.GlobalAveragePooling2D())

  return finalize(model)


def get_cnn_1d(archi_string):
  model = get_initial_model()

  from fc_core import th

  for i, c in enumerate(archi_string.split('-')):
    if c == 'p':
      model.add(mu.MaxPool1D(pool_size=2, strides=2))
      continue

    c = int(c)
    model.add(mu.HyperConv1D(
      filters=c, kernel_size=th.kernel_size, use_bias=False,
      activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))

  # model.add(mu.GlobalAveragePooling1D())
  model.add(mu.Flatten())

  return finalize(model)


def get_mlp(archi_string):
  from fc_core import th

  model = get_container(True)

  for n in archi_string.split('-'):
    model.add(mu.HyperDense(int(n), activation=th.activation,
                            layer_normalization=th.use_layernorm))
    if th.dropout > 0: model.add(mu.Dropout(1 - th.dropout))

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
  pass

