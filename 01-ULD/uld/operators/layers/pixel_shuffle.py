from tframe import tf
from tframe.layers import Input
from tframe.layers.layer import Layer


class PixelShuffle(Layer):

  abbreviation = 'ps'
  full_name = 'pixel_shuffle'


  def __init__(self, num):
    self.num = num

  def _link(self, x):
    bsize, s, h, w, c = x.shape
    bsize = x.shape[0]
    r = self.num

    y = tf.reshape(x, (bsize, h, w, r, r))
    y = tf.split(y, w, 1)
    y = tf.concat([tf.squeeze(i, axis=1) for i in y], 2)
    y = tf.split(y, h, 1)
    y = tf.concat([tf.squeeze(i, axis=1) for i in y], 2)

    return y
