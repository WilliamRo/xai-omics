from tframe.core.quantity import Quantity


def get_pw_rmse(epsilon=0.01):
  from tframe import tf
  """SET th.batch_size=1 ?"""
  def pw_rmse(truth, output):
    # [bs, num_slides, 440, 440, 1]
    axis = list(range(1, len(truth.shape)))
    a = tf.square(truth - output)
    b = tf.square(truth) + epsilon
    return tf.sqrt(tf.reduce_mean(a / b, axis=axis))

  return Quantity(pw_rmse, tf.reduce_mean, name='PW_RMSE', lower_is_better=True)


def get_ssim_3D():
  from tframe import tf
  def ssim(truth, output):
    # [bs, num_slides, 440, 440, 1]
    from uld_core import th
    if th.color_map is not None:
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
