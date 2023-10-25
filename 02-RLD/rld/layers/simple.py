from tframe import tf
from tframe.layers.layer import Layer, single_input


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
