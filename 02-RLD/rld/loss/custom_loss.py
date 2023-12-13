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


def get_relative_loss():
  from tframe import tf

  def relative_loss(truth, output):
    axis = list(range(1, len(truth.shape)))
    a = tf.abs(truth - output) / (tf.maximum(truth, output) + 1e-8)
    return tf.reduce_mean(a, axis=axis)

  return Quantity(relative_loss, tf.reduce_mean, name='Rela-Loss', lower_is_better=True)