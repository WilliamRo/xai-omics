from tframe import tf
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

import numpy as np



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


if __name__ == '__main__':
  import numpy as np
  # Put this line before importing tensorflow to get rid of future warnings
  from tframe import console, tf

  console.suppress_logging()
  console.start('Template')
  tf.InteractiveSession()

  k = 50

  np_x = np.array([0.05, 0.2, 0.8], dtype=float)
  print(f'np_x = {np_x}')

  np_h1 = np.tanh(np_x * k)
  print(np_h1)
  np_h2 = np.arctanh(np_h1) / k
  print(np_h2)

  x = tf.constant(np_x, dtype=tf.float32)

  h1 = Tanh_k(k)(x)
  console.eval_show(h1)

  h2 = tf.atanh(h1) / k
  console.eval_show(h2)
  console.end()
