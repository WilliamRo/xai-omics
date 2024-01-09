from tframe import context
from tframe.core import Function
from tframe.core.quantity import Quantity




def get_ssim_3D():
  from tframe import tf
  def ssim(truth, output):
    # [bs, num_slides, 440, 440, 1]
    from rld_core import th
    shape = [-1] + th.data_shape + [1]
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



def relative_loss(truth, output):
  from tframe import tf
  axis = list(range(1, len(truth.shape)))
  a = tf.abs(truth - output) / (tf.maximum(truth, output) + 1e-8)
  return tf.reduce_mean(a, axis=axis)


def get_relative_loss():
  from tframe import tf

  return Quantity(relative_loss, tf.reduce_mean, name='Rela-Loss', lower_is_better=True)


def get_internal_rela_loss(model):
  from tframe import tf

  internal_layer: Function = context.depot['unet']
  y = internal_layer.output_tensor

  alpha = tf.get_variable('internal_alpha', dtype=tf.float32,
                          initializer=10.0, trainable=False)

  def set_internal_alpha(value):
    v_plchd = tf.placeholder(dtype=tf.float32, name='alpha_place_holder')
    op = tf.assign(alpha, v_plchd)
    model.agent.session.run(op, feed_dict={v_plchd: value})
  context.depot['set_internal_alpha'] = set_internal_alpha

  rela_loss = relative_loss()
