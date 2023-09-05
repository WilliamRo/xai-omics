from tframe import console
from tframe import mu

import numpy as np



def get_initial_model():
  from mi_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from mi_core import th
  assert isinstance(model, mu.Predictor)

  model.add(mu.HyperConv3D(filters=1, kernel_size=1,
                           activation='sigmoid'))
  # Build model

  model.build(loss=get_loss(), metric=['loss', dice_acc()])
  # model.build(loss=get_loss(), metric='loss')
  return model


def get_loss():
  from tframe.core.quantity import Quantity
  from tframe import tf
  from tframe.losses import cross_entropy

  kernel, tf_summ_method, np_summ_method = None, tf.reduce_mean, None

  def _cross_entropy(labels, outputs):
    epsilon = 0.001
    epsilon_tensor = tf.fill(tf.shape(labels), epsilon)
    results = tf.multiply(tf.abs(labels - outputs),
                          tf.add(labels, epsilon_tensor))

    return results * 100


  def _dice_coef_loss(labels, outputs):
    smooth = 1.0

    # flatten
    labels_f = tf.layers.flatten(labels)
    outputs_f = tf.layers.flatten(outputs)

    intersection = tf.reduce_sum(labels_f * outputs_f, axis=1)
    dice_coef = ((2.0 * intersection + smooth) /
                 (tf.reduce_sum(labels_f, axis=1) +
                  tf.reduce_sum(outputs_f, axis=1) + smooth))
    dice_coef = tf.expand_dims(dice_coef, 1)

    return 1.0 - dice_coef

  use_logits = False
  kernel = _dice_coef_loss
  # kernel = _cross_entropy


  return Quantity(kernel, tf_summ_method, np_summ_method,
                  use_logits=use_logits, lower_is_better=True,
                  name='dice_coef_loss')


def dice_acc():
  from tframe.core.quantity import Quantity
  from tframe import tf
  from tframe.losses import cross_entropy

  kernel, tf_summ_method, np_summ_method = None, tf.reduce_mean, None

  def _dice_accuracy(labels, outputs):
    smooth = 1.0

    # Flatten
    labels_f = tf.keras.layers.Flatten()(labels)
    outputs_f = tf.keras.layers.Flatten()(outputs)

    # Discretization
    outputs_f = tf.where(outputs_f > 0.5, tf.ones_like(outputs_f),
                         tf.zeros_like(outputs_f))

    intersection = tf.reduce_sum(labels_f * outputs_f, axis=1)
    dice_coef = ((2.0 * intersection + smooth) /
                 (tf.reduce_sum(labels_f, axis=1) +
                  tf.reduce_sum(outputs_f, axis=1) + smooth))
    dice_coef = tf.expand_dims(dice_coef, 1)

    return dice_coef

  use_logits = False
  kernel = _dice_accuracy

  return Quantity(kernel, tf_summ_method, np_summ_method,
                  use_logits=use_logits, lower_is_better=False,
                  name='dice_accuracy')


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_cnn():
  model = get_initial_model()
  from mi_core import th

  fm = mu.ForkMergeDAG(
    [mu.Conv3D(64, 3, activation='relu'),  #1
     mu.Conv3D(2, 3, activation='relu'),  #2
     mu.Conv3D(32, 3, activation='relu'),  #3
     mu.Conv3D(16, 3, activation='relu'),  #4
     mu.Conv3D(8, 3, activation='relu'),  #5
     mu.Conv3D(4, 3, activation='relu'),  #6
     mu.Conv3D(2, 3, activation='relu'),  #7
     mu.Merge.Sum(),  #8
     mu.Conv3D(1, 3)],                              #9
    edges='1;01;010;0001;00001;000001;0000001;00100001;000000001',
    name='simple_cnn'
  )
  model.add(fm)

  return finalize(model)



if __name__ == '__main__':
  # ++ Blue box for model
  from mi_core import th

