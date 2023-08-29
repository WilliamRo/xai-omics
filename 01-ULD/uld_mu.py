from tframe.core.quantity import Quantity
from tframe import console
from tframe import mu
from uld.operators.custom_layers import Tanh_k, Atanh_k

import numpy as np



def get_initial_model():
  from uld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.use_tanh != 0:
    model.add(Tanh_k(k=th.use_tanh))
  return model


def finalize(model):
  from uld_core import th
  from tframe import tf

  assert isinstance(model, mu.Predictor)
  model.add(mu.HyperConv3D(filters=1, kernel_size=1))
  # model.add(mu.Activation('sigmoid'))



  if th.learn_delta:
    model.input_.abbreviation = 'input'
    model.add(mu.ShortCut(model.input_, mode=mu.ShortCut.Mode.SUM))
    # model.add(mu.Activation('lrelu'))

  if th.use_tanh != 0:
    model.add(Atanh_k(k=th.use_tanh))
  # else:
  #   model.add(mu.Activation('sigmoid'))
  # Build model
  # model.build(loss=th.loss_string, metric=['loss'])
  # model.build(loss=th.loss_string, metric=[get_ssim_3D(), 'loss'])
  model.build(loss=th.loss_string, metric=[
    get_ssim_3D(), get_nrmse(), get_psnr(), get_pw_rmse(), 'loss'])
  # model.build(loss=get_pw_rmse(), metric=[
  #   get_ssim_3D(), get_nrmse(), get_psnr(), 'loss'])
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()
  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_pw_rmse():
  from tframe import tf
  """SET th.batch_size=1 ?"""
  def pw_rmse(truth, output):
    # [bs, num_slides, 440, 440, 1]
    axis = list(range(1, len(truth.shape)))
    a = tf.square(truth - output)
    b = tf.square(truth) + 0.001
    return tf.sqrt(tf.reduce_mean(a / b, axis=axis))

  return Quantity(pw_rmse, tf.reduce_mean, name='PW_RMSE', lower_is_better=True)


def get_ssim_3D():
  from tframe import tf
  def ssim(truth, output):
    # [bs, num_slides, 440, 440, 1]
    from uld_core import th
    if th.use_color:
      shape = [-1, 440, 440, 3]
    else:
      shape = [-1, 440, 440, 1]
    truth, output = [tf.reshape(x, shape) for x in (truth, output)]

    return tf.image.ssim(truth, output, max_val=1.0)

  return Quantity(ssim, tf.reduce_mean, name='SSIM', lower_is_better=False)


def get_nrmse():
  """SET th.[e]val_batch_size=1"""
  from tframe import tf

  def nrmse(truth, output):
    # [bs, num_slides, 440, 440, 1]
    axis = list(range(1, len(truth.shape)))
    a = tf.reduce_sum(tf.square(truth - output), axis=axis)
    b = tf.reduce_sum(tf.square(truth), axis=axis)
    return tf.sqrt(a / b)

  return Quantity(nrmse, tf.reduce_mean, name='NRMSE', lower_is_better=True)


def get_psnr():
  from tframe import tf

  def psnr(truth, output):
    # [bs, num_slides, 440, 440, 1]
    return tf.image.psnr(truth, output, 1)

  return Quantity(psnr, tf.reduce_mean, name='PSNR', lower_is_better=False)
