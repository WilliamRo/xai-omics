from tframe import tf
from tframe import hub as th

from tframe.layers.common import Input
from tframe.layers.layer import Layer



class NormalizeEnergy(Layer):

  abbreviation = 'ne'
  full_name = 'norm_energy'

  def __init__(self, input_layer, gamma=0):
    self.input_layer: Input = input_layer
    self.gamma = gamma


  def _link(self, x):
    x = tf.abs(x)
    # x.shape = [?, S, H, W, 1]

    model_input = self.input_layer.output_tensor

    e = tf.reduce_sum(model_input, axis=[1, 2, 3, 4])

    if self.gamma > 0:
      _g = tf.get_variable('gamma', shape=(1,), dtype=th.dtype,
                           initializer=tf.initializers.zeros())
      _g = tf.nn.sigmoid(_g)
      e = e / (1 + self.gamma * _g)

    return x / tf.reduce_sum(x, axis=[1, 2, 3, 4]) * e






