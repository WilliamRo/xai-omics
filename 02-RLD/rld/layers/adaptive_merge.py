from tframe import tf
from tframe import hub as th

from tframe.layers.common import Input
from tframe.layers.layer import Layer, single_input
from tframe.layers.merge import Merge
from tframe.operators.neurons import NeuroBase




class WeightedSum(Merge):

  abbreviation = 'ws'
  full_name = 'weighted_sum'

  def __init__(self): pass


  def _link(self, xs):
    assert len(xs) == 2

    # xs[1] should be weights
    th.depot['weight_map'] = xs[1]
    th.depot['candidate1'] = xs[0]

    # x.shape = [?, S, H, W, C]
    y = xs[0] * xs[1]
    y = tf.reduce_sum(y, axis=-1, keep_dims=True)
    # y.shape = [?, S, H, W, 1]
    return y



class Highlighter(Layer):

  abbreviation = 'high'
  full_name = 'highlighter'

  def __init__(self, beta):
    self.beta = beta
    self.abbreviation += f'{beta}'
    self.full_name += f'{beta}'

  @single_input
  def _link(self, x):
    rescaled_x = tf.clip_by_value(x / self.beta, 0, clip_value_max=1.0)
    y = tf.concat([x, rescaled_x], axis=-1)

    return y