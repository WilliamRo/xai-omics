from tframe.layers import Reshape
from tframe.layers.layer import single_input, Layer
from tframe import tf
from uld.operators.layers.adaptive_merge import *
from uld.operators.layers.energy_norm import *
from uld.operators.layers.guass_pyramid import *


class Tanh_k(Layer):

  def __init__(self, k: float):
    self.k = k

    self.full_name = f'Tanh_{k}'
    self.abbreviation = self.full_name

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    y = tf.tanh(self.k * x)
    return y


class Atanh_k(Layer):

  def __init__(self, k: float):
    self.k = k

    self.full_name = f'Atanh_{k}'
    self.abbreviation = self.full_name

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # epsilon = 1e-6
    MAX_I = np.tanh(1.0 * self.k)
    x = tf.sigmoid(x) * MAX_I
    y = tf.atanh(x) / self.k
    return y


class Clip(Layer):

  def __init__(self, vmin: float, vmax: float):
    self.min = vmin
    self.max = vmax

    self.full_name = f'Clip_{vmin}_{vmax}'
    self.abbreviation = self.full_name

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    y = tf.clip_by_value(x, self.min, self.max)
    return y


class Flatten(Reshape):
# todo: it doesnt work
  @single_input
  def _link(self, input_, **kwargs):
    output_shape = [-1, tf.dimension_value(None)]
    return tf.reshape(input_, output_shape)




if __name__ == '__main__':
  import numpy as np
  # Put this line before importing tensorflow to get rid of future warnings
  from tframe import console, tf

  console.suppress_logging()
  console.start('Template')
  tf.InteractiveSession()

  k = 50

  np_x = np.array([np.NaN, -5.2, 0.2, 0.8, 5.7, np.Inf], dtype=float)
  # print(f'np_x = {np_x}')
  #
  # np_h1 = np.tanh(np_x * k)
  # print(np_h1)
  # np_h2 = np.arctanh(np_h1) / k
  # print(np_h2)

  x = tf.constant(np_x, dtype=tf.float32)
  h1 = Clip(0, 1.2)(x)
  # h1 = Tanh_k(k)(x)
  console.eval_show(h1)


  # h2 = tf.atanh(h1) / k
  # console.eval_show(h2)
  console.end()
