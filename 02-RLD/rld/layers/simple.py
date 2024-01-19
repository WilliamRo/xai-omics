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


class ChannelSplit(Layer):

  def __init__(self, channel: int):
    self.channel = channel

    self.full_name = f'Channel_{channel}_Split'
    self.abbreviation = self.full_name

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    return x[:, ..., self.channel:self.channel + 1]

