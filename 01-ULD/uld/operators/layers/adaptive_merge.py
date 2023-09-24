from tframe import tf
from tframe import hub as th

from tframe.layers.common import Input
from tframe.layers.layer import Layer, single_input
from tframe.layers.merge import Merge
from tframe.operators.neurons import NeuroBase



class AdaptiveMerge(Merge, NeuroBase):

  abbreviation = 'am'
  full_name = 'adaptive_merge'

  def __init__(self, input_layer):
    NeuroBase.__init__(self)
    self.input_layer: Input = input_layer


  def _link(self, xs):
    # Each x, including input, has a shape of [?, S, H, W, 1]
    N = len(xs)

    # Weights will be calculated based on low dose input
    low = self.input_layer.output_tensor

    h1 = tf.reduce_max(low, axis=[2, 3, 4])
    h2 = tf.reduce_mean(low, axis=[2, 3, 4])
    # h[12].shape = [?, S]
    h = tf.stack([h1, h2], axis=-1)
    # h.shape = [?, S, 2]
    h = self.dense(2, h, 'h1', activation='relu')
    h = self.dense(2, h, 'h2', activation='relu')

    # ws.shape = [?, S, N]
    ws = self.dense(N, h, 'ws', activation='softmax')
    ws = tf.split(ws, N, axis=-1)
    # each w.shape = [?, S, 1]

    ws = [tf.expand_dims(w, -1) for w in ws]
    ws = [tf.expand_dims(w, -1) for w in ws]
    # each w.shape = [?, S, 1, 1, 1]

    return tf.add_n([w * x for w, x in zip(ws, xs)])



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