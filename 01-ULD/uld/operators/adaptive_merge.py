from tframe import tf
from tframe import hub as th

from tframe.layers.common import Input
from tframe.layers.layer import Layer
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
