from tframe import console
from tframe import mu
from tframe import tf
from tframe.core import init_with_graph
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

import numpy as np



def get_initial_model():
  from mic_core import th

  model = mu.Classifier(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from mic_core import th
  assert isinstance(model, mu.Classifier)

  model.add(GlobalAveragePooling3D())

  # Build model
  model.add(mu.Activation('softmax'))

  model.build(batch_metric='accuracy')
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_cnn_3d():
  model = get_initial_model()

  fm = mu.ForkMergeDAG(
    [mu.HyperConv3D(128, 3, activation='relu'),  #1
     mu.HyperConv3D(64, 3, activation='relu'),           #2
     mu.HyperConv3D(32, 3, activation='relu'),           #3
     mu.HyperConv3D(16, 3, activation='relu'),           #4
     mu.HyperConv3D(8, 3, activation='relu'),            #5
     mu.HyperConv3D(4, 3, activation='relu'),            #6
     mu.HyperConv3D(2, 3, activation='relu'),            #7
     mu.HyperConv3D(1, 3)],                              #8
    edges='1;01;001;0001;00001;000001;0000001;00000001',
    name='cnn_3d'
  )
  model.add(fm)

  return finalize(model)


def get_fcn_3d_01():
  model = get_initial_model()

  fm = mu.ForkMergeDAG([
    mu.HyperConv3D(filters=3, kernel_size=3, activation='relu'),       #1
    mu.MaxPool3D(pool_size=2, strides=2, padding='valid'),                   #2
    mu.HyperConv3D(filters=16, kernel_size=3, activation='relu'),      #3
    mu.MaxPool3D(pool_size=2, strides=2, padding='valid'),                   #4
    mu.HyperConv3D(filters=32, kernel_size=3, activation='relu'),      #5
    mu.MaxPool3D(pool_size=2, strides=2, padding='valid'),                   #6
    mu.HyperConv3D(filters=64, kernel_size=3, activation='relu'),      #7
    mu.MaxPool3D(pool_size=2, strides=2, padding='valid'),                   #8
    mu.HyperDeconv3D(filters=32, kernel_size=3, strides=2, activation='relu'),    #9
    mu.Merge.Sum(),                                                                    #10
    mu.HyperDeconv3D(filters=2, kernel_size=3,  strides=8, activation='relu')],#11
    edges=
    '1;01;001;0001;00001;000001;0000001;00000001;000000001;0000001001;00000000001',
    name='fcn_3d')
  model.add(fm)

  return finalize(model)


def get_fcn_3d_02():
  model = get_initial_model()
  stride = (1, 2, 2)

  fm = mu.ForkMergeDAG([
    mu.HyperConv3D(filters=3, kernel_size=3, activation='relu'),       #1
    mu.MaxPool3D(pool_size=stride, strides=stride, padding='valid'),                   #2
    mu.HyperConv3D(filters=16, kernel_size=3, activation='relu'),      #3
    mu.MaxPool3D(pool_size=stride, strides=stride, padding='valid'),                   #4
    mu.HyperConv3D(filters=32, kernel_size=3, activation='relu'),      #5
    mu.MaxPool3D(pool_size=stride, strides=stride, padding='valid'),                   #6
    mu.HyperConv3D(filters=64, kernel_size=3, activation='relu'),      #7
    mu.MaxPool3D(pool_size=stride, strides=stride, padding='valid'),                   #8
    mu.HyperDeconv3D(filters=32, kernel_size=3, strides=stride, activation='relu'),    #9
    mu.Merge.Sum(),                                                                    #10
    mu.HyperDeconv3D(filters=2, kernel_size=3,  strides=(1, 8, 8), activation='relu')],#11
    edges=
    '1;01;001;0001;00001;000001;0000001;00000001;000000001;0000001001;00000000001',
    name='fcn_3d')
  model.add(fm)

  return finalize(model)


def get_fcn_3d_03():
  model = get_initial_model()
  stride = (1, 2, 2)

  fm = mu.ForkMergeDAG([
    mu.HyperConv3D(filters=3, kernel_size=3, strides=stride, activation='relu'),       #1
    mu.HyperConv3D(filters=16, kernel_size=3, strides=stride, activation='relu'),      #3
    mu.HyperConv3D(filters=32, kernel_size=3, strides=stride, activation='relu'),      #5
    mu.HyperConv3D(filters=64, kernel_size=3, strides=stride, activation='relu'),      #7
    mu.HyperDeconv3D(filters=32, kernel_size=3, strides=stride, activation='relu'),    #9
    mu.Merge.Sum(),                                                                    #10
    mu.HyperDeconv3D(filters=2, kernel_size=3,  strides=(1, 8, 8), activation='relu')],#11
    edges='1;01;001;0001;00001;000101;0000001', name='fcn_3d')
  model.add(fm)

  return finalize(model)


def get_fcn_3d_04():
  model = get_initial_model()
  stride = (1, 2, 2)

  fm = mu.ForkMergeDAG([
    mu.HyperConv3D(filters=3, kernel_size=3, strides=stride, activation='relu'),       #1
    mu.HyperConv3D(filters=16, kernel_size=3, strides=stride, activation='relu'),      #3
    mu.HyperConv3D(filters=32, kernel_size=3, strides=stride, activation='relu'),      #5
    mu.HyperConv3D(filters=64, kernel_size=3, strides=stride, activation='relu'),      #7
    mu.HyperDeconv3D(filters=32, kernel_size=3, strides=stride, activation='relu'),    #9
    mu.Merge.Sum(),                                                                    #10
    mu.HyperConv3D(filters=2, kernel_size=1, strides=1, activation='relu')],#11
    edges='1;01;001;0001;00001;000101;0000001', name='fcn_3d')
  model.add(fm)

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


