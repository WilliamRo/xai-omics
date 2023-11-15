from tframe import console, context
from tframe import mu
from tframe import tf
from tframe.core import init_with_graph
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from copy import deepcopy
from tframe.core import TensorSlot

import numpy as np



# Model
def get_initial_model():
  from es_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model, add_last_layer=True):
  assert isinstance(model, mu.Predictor)

  if add_last_layer: model.add(
    mu.HyperConv3D(filters=1, kernel_size=1, activation='sigmoid'))

  # Build model
  from es_core import th
  metric = ['loss', dice_acc()] if not th.if_pre_train else [dice_acc()]

  model.build(loss=get_loss(), metric=metric)

  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()

  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_cnn():
  model = get_initial_model()
  from es_core import th

  fm = mu.ForkMergeDAG(
    [mu.HyperConv3D(16, 3, activation='relu', use_bias=True),  #1
     mu.HyperConv3D(8, 3, activation='relu', use_bias=True),  #2
     mu.HyperConv3D(4, 3, activation='relu', use_bias=True)],  #3
    edges='1;01;001',
    name='simple_cnn'
  )
  model.add(fm)

  return finalize(model)


def get_ynet_1(arc_string='8-5-2-3-lrelu-mp'):
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
  for i in range(height):
    for _ in range(width):
      encoder.append(conv(filters * 2**i, kernel_size, activation))
    if use_maxpool: encoder.append(mp())

  for _ in range(width):
    encoder.append(conv(filters * 2 ** height, kernel_size, activation))

  # Decoder Setting
  decoder_region = []
  for i in range(height):
    filters_index = height - i - 1
    if use_maxpool: decoder_region.append(
        deconv(filters * 2 ** filters_index, kernel_size, activation, 2))
    for _ in range(width):
      decoder_region.append(
        conv(filters * 2 ** filters_index, kernel_size, activation))
  decoder_region.append(conv(1, kernel_size, 'sigmoid'))

  decoder_lesion = deepcopy(decoder_region)

  vertice = [
    encoder,
    decoder_region,   # yield y_r
    decoder_lesion,
    mu.Merge(mu.Merge.PROD),
  ]
  edges = '1;01;010;0011'

  model.add(mu.ForkMergeDAG(vertice, edges, name='y-net'))

  context.depot['region_layer'] = vertice[1][-1]

  def get_region_loss(_):
    from tframe import pedia
    from tframe.core.function import Function
    from es_core import th

    region_layer: Function = context.depot['region_layer']
    y_r = region_layer.output_tensor

    m_r = tf.placeholder(dtype=th.dtype, name='region_mask')
    tf.add_to_collection(pedia.default_feed_dict, m_r)

    alpha = th.alpha
    rl = tf.reduce_mean(y_r * (1 - m_r))

    return [alpha * rl]

  context.customized_loss_f_net = get_region_loss

  return finalize(model, add_last_layer=False)


def get_ynet_2(arc_string='8-5-2-3-lrelu-mp'):
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

  # Construct encoder
  encoder = []
  for i in range(height + 1):
    if i != 0: encoder.append([mp()])
    encoder.append(
      [conv(filters * 2**i, kernel_size, activation) for _ in range(width)])

  # Construct decoder region
  decoder_region = []
  for i in range(height):
    f_index = height - i - 1
    if use_maxpool:
      decoder_region.append(
        [deconv(filters * 2 ** f_index, kernel_size, activation, 2)])
      decoder_region.append([mu.Merge(mu.Merge.CONCAT)])
      decoder_region.append([conv(
        filters * 2 ** f_index, kernel_size, activation) for _ in range(width)])
  decoder_region.append([
    mu.HyperConv3D(1, kernel_size, activation='sigmoid', use_bias=True,
                   bias_initializer=th.bias_initializer)])

  # Construct decoder lesion
  decoder_lesion = []
  for i in range(height):
    f_index = height - i - 1
    if use_maxpool:
      decoder_lesion.append(
        [deconv(filters * 2 ** f_index, kernel_size, activation, 2)])
      decoder_lesion.append([mu.Merge(mu.Merge.CONCAT)])
      decoder_lesion.append([conv(
        filters * 2 ** f_index, kernel_size, activation) for _ in range(width)])
  decoder_lesion.append([conv(1, kernel_size, 'sigmoid')])

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

  # context setting
  context.depot['region_layer'] = vertice[
    encoder_len + decoder_len - 1][-1]
  context.depot['lesion_layer'] = vertice[
    encoder_len + decoder_len * 2 - 1][-1]

  # extra loss setting
  context.customized_loss_f_net = get_total_loss

  return finalize(model, add_last_layer=False)


def get_ynet_3(arc_string='8-5-2-3-lrelu-mp'):
  from es_core import th

  model = get_initial_model()

  # Parameter Setting
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

  # Construct encoder
  encoder = []
  for i in range(height + 1):
    if i != 0: encoder.append([mp()])
    encoder.append(
      [conv(filters * 2**i, kernel_size, activation) for _ in range(width)])

  # Construct decoder region
  decoder_region = []
  for i in range(height):
    f_index = height - i - 1
    if use_maxpool:
      decoder_region.append(
        [deconv(filters * 2 ** f_index, kernel_size, activation, 2)])
      decoder_region.append([conv(
        filters * 2 ** f_index, kernel_size, activation) for _ in range(width)])
  decoder_region.append([
    mu.HyperConv3D(1, kernel_size, activation='sigmoid', use_bias=True,
                   bias_initializer=th.bias_initializer)])

  # Construct decoder lesion
  decoder_lesion = []
  for i in range(height):
    f_index = height - i - 1
    if use_maxpool:
      decoder_lesion.append(
        [deconv(filters * 2 ** f_index, kernel_size, activation, 2)])
      decoder_lesion.append([mu.Merge(mu.Merge.CONCAT)])
      decoder_lesion.append([conv(
        filters * 2 ** f_index, kernel_size, activation) for _ in range(width)])
  decoder_lesion.append([conv(1, kernel_size, 'sigmoid')])

  # Vertices
  vertice = encoder + decoder_region + decoder_lesion + [[mu.Merge(mu.Merge.PROD)]]
  conv_index = [
    vertice.index(l) - 1 for l in vertice if isinstance(l[0], mu.MaxPool3D)]
  concat_index = [
    vertice.index(l) for l in vertice if l[0].full_name == 'concat']
  concat_index_lesion = concat_index

  encoder_len = len(encoder)
  decoder_region_len = len(decoder_region)
  decoder_lesion_len = len(decoder_lesion)

  # Edges
  edge = []
  for i in range(len(vertice)):
    if i < encoder_len:
      # Encoder
      edge.append('0' * i + '1')
    elif i < encoder_len + decoder_region_len:
      # Decoder Region
      e = '0' * i + '1'
      edge.append(e)
    elif i == len(vertice) - 1:
      # Prod
      e = '0' * (encoder_len + decoder_region_len) + '1' + '0' * (decoder_lesion_len - 1) + '1'
      edge.append(e)
    else:
      # Decoder Lesion
      if i == encoder_len + decoder_region_len:
        e = '0' * encoder_len + '1' + '0' * decoder_region_len
      elif i in concat_index_lesion:
        index = concat_index_lesion.index(i)
        index = conv_index[-(index + 1)] + 1
        e = '0' * index + '1' + '0' * (i - index - 1) + '1'
      else:
        e = '0' * i + '1'
      edge.append(e)


  edges = ';'.join(edge)

  model.add(mu.ForkMergeDAG(vertice, edges, name='y-net'))

  # context setting
  context.depot['region_layer'] = vertice[
    encoder_len + decoder_region_len - 1][-1]
  context.depot['lesion_layer'] = vertice[
    encoder_len + decoder_region_len + decoder_lesion_len - 1][-1]

  # extra loss setting
  context.customized_loss_f_net = get_total_loss

  return finalize(model, add_last_layer=False)


# Loss
def get_loss():
  from tframe.core.quantity import Quantity
  from tframe import tf
  from es_core import th
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

    return th.alpha_total * 1.0 * (1.0 - dice_coef)


  use_logits = False
  kernel = _dice_coef_loss
  # kernel = _combined_loss

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


def get_region_loss(model):
  from tframe import pedia
  from tframe.core.function import Function
  from es_core import th

  region_layer: Function = context.depot['region_layer']
  y_r = region_layer.output_tensor

  m_r = tf.placeholder(dtype=th.dtype, name='region_mask')
  tf.add_to_collection(pedia.default_feed_dict, m_r)

  alpha = th.alpha_region
  region_loss = tf.reduce_mean(y_r * (1 - m_r))

  l_r_slot = TensorSlot(model, 'Loss_region')
  model._update_group.add(l_r_slot)

  l_r = tf.multiply(alpha * 1.0, region_loss, name='region_loss')
  l_r_slot.plug(l_r)

  return l_r


def get_lesion_loss(model):
  from tframe import pedia
  from tframe.core.function import Function
  from es_core import th

  lesion_layer: Function = context.depot['lesion_layer']
  y_r = lesion_layer.output_tensor

  m_r = tf.placeholder(dtype=th.dtype, name='lesion_mask')
  tf.add_to_collection(pedia.default_feed_dict, m_r)

  alpha = th.alpha_lesion
  smooth = 1.0

  # flatten
  labels_f = tf.layers.flatten(m_r)
  outputs_f = tf.layers.flatten(y_r)

  intersection = tf.reduce_sum(labels_f * outputs_f, axis=1)
  dice_coef = ((2.0 * intersection + smooth) /
               (tf.reduce_sum(labels_f, axis=1) +
                tf.reduce_sum(outputs_f, axis=1) + smooth))
  lesion_loss = tf.reduce_mean(1.0 - dice_coef)

  l_l_slot = TensorSlot(model, 'Loss_lesion')
  model._update_group.add(l_l_slot)

  l_l = tf.multiply(alpha * 1.0, lesion_loss, name='lesion_loss')

  l_l_slot.plug(l_l)

  return l_l


def get_total_loss(_):
  return [get_region_loss(_), get_lesion_loss(_)]



if __name__ == '__main__':
  # ++ Blue box for model
  from es_core import th

