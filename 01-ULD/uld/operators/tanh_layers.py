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
