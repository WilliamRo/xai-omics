from tframe import console
from tframe import mu
from copy import deepcopy

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



def get_ynet(arc_string='8-5-2-3-lrelu-mp'):
  from es_core import th

  model = get_initial_model()

  use_maxpool = False
  option = arc_string.split('-')
  filters, kernel_size, height, width = [int(op) for op in option[:4]]
  activation = option[4]
  for op in option[5:]:
    if op in ('mp', 'maxpool'): use_maxpool = True

  # Layer Setting
  conv = lambda _c, _ks, _act: mu.HyperConv3D(
    _c, _ks, activation=_act)
  deconv = lambda _c, _ks, _act, _s: mu.HyperDeconv3D(
    _c, _ks, activation=_act, strides=_s)
  mp = lambda _ps=2, _s=2: mu.MaxPool3D(_ps, _s)

  # Encoder Setting
  encoder = []
  for i in range(height + 1):
    if i != 0: encoder.append([mp()])
    encoder.append(
      [conv(filters * 2**i, kernel_size, activation) for _ in range(width)])

  # Decoder Setting
  decoder_region = []
  for i in range(height):
    f_index = height - i - 1
    if use_maxpool:
      decoder_region.append(
        [deconv(filters * 2 ** f_index, kernel_size, activation, 2)])
      decoder_region.append([mu.Merge(mu.Merge.CONCAT)])
      decoder_region.append([conv(
        filters * 2 ** f_index, kernel_size, activation) for _ in range(width)])
  decoder_region.append([conv(1, kernel_size, 'sigmoid')])

  decoder_lesion = deepcopy(decoder_region)

  # Vertice and Edges
  vertice = encoder + decoder_region + decoder_lesion + [[mu.Merge(mu.Merge.PROD)]]
  conv_index = [
    vertice.index(l) - 1 for l in vertice if isinstance(l[0], mu.MaxPool3D)]
  concat_index = [
    vertice.index(l) for l in vertice if isinstance(l[0], mu.Merge)]
  concat_index_region = concat_index[:len(concat_index) // 2]
  concat_index_lesion = concat_index[len(concat_index) // 2:]

  encoder_len = len(encoder)
  decoder_len = len(decoder_region)
  edge = []
  for i in range(len(vertice)):
    if i < encoder_len:
      edge.append('0' * i + '1')
    elif i < encoder_len + decoder_len:
      if i in concat_index_region:
        index = concat_index_region.index(i)
        index = conv_index[-(index + 1)] + 1
        e = '0' * index + '1' + '0' * (i - index - 1) + '1'
      else:
        e = '0' * i + '1'
      edge.append(e)
    elif i == len(vertice) - 1:
      e = '0' * (encoder_len + decoder_len) + '1' + '0' * (decoder_len - 1) + '1'
      edge.append(e)
    else:
      if i == encoder_len + decoder_len:
        e = '0' * encoder_len + '1' + '0' * decoder_len
      elif i in concat_index_lesion:
        index = concat_index_lesion.index(i)
        index = conv_index[-(index + 1)] + 1
        e = '0' * index + '1' + '0' * (i - index - 1) + '1'
      else:
        e = '0' * i + '1'
      edge.append(e)


  edges = ';'.join(edge)

  model.add(mu.ForkMergeDAG(vertice, edges, name='y-net'))

  return finalize(model)



if __name__ == '__main__':
  # ++ Blue box for model
  from mi_core import th

