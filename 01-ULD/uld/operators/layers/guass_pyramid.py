from tframe import tf
from tframe import hub as th

from tframe.layers.common import Input
from tframe.layers.layer import Layer



class GaussianPyramid3D(Layer):

  abbreviation = 'gp'
  full_name = 'gaussian_pyramid'

  def __init__(self, kernel_size, sigmas):
    self.kernel_size = kernel_size
    self.sigmas = sigmas


  def gen_gauss_kernels(self):
    ks = self.kernel_size
    ax = tf.range(-ks // 2 + 1.0, ks // 2 + 1.0)
    xx, yy, zz = tf.meshgrid(ax, ax, ax)

    kernels = []
    for sigma in self.sigmas:
      kernel = tf.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2.0 * sigma ** 2))
      kernel = kernel / tf.reduce_sum(kernel)
      # kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 1, 1])
      kernel = kernel[..., tf.newaxis, tf.newaxis]
      kernels.append(kernel)
    return kernels


  def _link(self, x):
    # TODO: ???????
    if isinstance(x, list): x = x[0]
    # x.shape = [?, S, H, W, 1]
    ys = []
    for k in self.gen_gauss_kernels():
      y = tf.nn.conv3d(x, k, strides=[1, 1, 1, 1, 1], padding='SAME')
      ys.append(y)

    return tf.concat(ys, axis=-1)






