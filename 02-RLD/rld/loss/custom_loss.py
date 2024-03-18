from tframe import context, tf
from tframe.core import Function, TensorSlot
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


def nrmse(truth, output):
  # [bs, num_slides, 440, 440, 1]
  axis = list(range(1, len(truth.shape)))
  a = tf.reduce_sum(tf.square(truth - output), axis=axis)
  b = tf.reduce_sum(tf.square(truth), axis=axis)
  return tf.sqrt(a / b)


def get_nrmse():
  """SET th.[e]val_batch_size=1"""
  from tframe import tf

  return Quantity(nrmse, tf.reduce_mean, name='NRMSE', lower_is_better=True)


def get_psnr():
  from tframe import tf

  def psnr(truth, output):
    # [bs, num_slides, 440, 440, 1]
    truth = tf.reduce_mean(truth, axis=1)
    output = tf.reduce_mean(output, axis=1)
    return tf.image.psnr(truth, output, 1)

  return Quantity(psnr, tf.reduce_mean, name='PSNR', lower_is_better=False)



def relative_loss(truth, output):
  from tframe import tf
  axis = list(range(1, len(output.shape)))
  a = tf.abs(truth - output) / (tf.maximum(truth, output) + 1e-8)
  res = tf.reduce_mean(a, axis=axis)
  return res




def get_relative_loss(name='Rela-Loss'):
  from tframe import tf
  from rld_core import th
  def out_relative_loss(truth, output):
    from tframe import tf
    axis = list(range(1, len(output.shape)))
    a = tf.abs(truth - output) / (tf.maximum(truth, output) + 1e-8)
    res = tf.reduce_mean(a, axis=axis)
    if th.internal_loss:
      res = tf.multiply(1-th.alpha, res)
    return res

  return Quantity(out_relative_loss, tf.reduce_mean, name=name,
                  lower_is_better=True)


def get_internal_rela_loss(model):
  from tframe import tf, pedia
  from rld_core import th

  internal_layer: Function = context.depot['unet']
  y_out = internal_layer.output_tensor

  # y_true = tf.placeholder(dtype=th.dtype, name='targets')
  # tf.add_to_collection(pedia.default_feed_dict, y_true)
  y_true = model._targets._op

  alpha = tf.get_variable('internal_alpha', dtype=tf.float32,
                          initializer=th.alpha, trainable=False)
  def set_internal_alpha(value):
    v_plchd = tf.placeholder(dtype=tf.float32, name='alpha_place_holder')
    op = tf.assign(alpha, v_plchd)
    model.agent.session.run(op, feed_dict={v_plchd: value})
  context.depot['set_internal_alpha'] = set_internal_alpha

  rela_loss = relative_loss(y_true, y_out)
  rela_loss = tf.reduce_mean(rela_loss)

  # rela_loss = _dice_coef_loss(y_true, y_out)

  l_r_slot = TensorSlot(model, 'internal loss')
  model._update_group.add(l_r_slot)

  l_r = tf.multiply(alpha, rela_loss, name='internal_loss')
  l_r_slot.plug(l_r)

  return l_r


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

    return (1.0 - dice_coef)


def get_total_loss(_):
  return [get_internal_rela_loss(_)]

if __name__ == '__main__':
  truth = tf.ones((2, 440, 440, 256, 1))
  output = tf.ones((2, 440, 440, 256, 1))

  loss = _dice_coef_loss(truth, output)

  print(loss)

