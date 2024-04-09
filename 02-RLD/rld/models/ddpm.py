import numpy as np

from tframe import hub as th, DataSet
from tframe.models import GaussianDiffusion


class DDPM(GaussianDiffusion):

  def __index__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def _sample_with_clip(self, x_t, t, pred_noise):
    # Calculate predicted image
    coef_shape = [-1] + [1] * (len(pred_noise.shape) - 1)
    alpha_t = self.alphas[t].reshape(coef_shape)
    bar_alpha_t = self.bar_alphas[t].reshape(coef_shape)
    beta_t = self.betas[t].reshape(coef_shape)

    # Predict x_0 using pred_noise
    x_0_pred = np.sqrt(1. / bar_alpha_t) * x_t - np.sqrt(
      1. / bar_alpha_t - 1) * pred_noise
    x_0_pred = np.clip(x_0_pred, -1., 1.)

    z = np.random.randn(x_t.shape[0], *th.data_shape[1:], 1)
    if t > 0:
      sqrt_bar_alpha_t_prev = self.sqrt_bar_alphas[t - 1].reshape(coef_shape)
      bar_alpha_t_prev = self.bar_alphas[t - 1].reshape(coef_shape)
      sqrt_alpha_t = self.sqrt_alphas[t].reshape(coef_shape)
      mean = (beta_t * sqrt_bar_alpha_t_prev) / ( 1. - bar_alpha_t) * x_0_pred
      mean += ((1. - bar_alpha_t_prev) * sqrt_alpha_t) / (1. - bar_alpha_t) * x_t
      std = np.sqrt(beta_t * (1. - bar_alpha_t_prev) / (1. - bar_alpha_t))
    else:
      mean = (beta_t / (1. - bar_alpha_t)) * x_0_pred
      std = 0.

    return mean + std * z

  def generate(self, sample_num=1, x_T=None, return_all_images=False,
               clip=True, delta_t=0):
    """See DDPM paper -> Algorithm 2
    Ref: https://github.com/bot66/MNISTDiffusion/blob/main/train_mnist.py
    """
    if x_T is None:
      x_t = np.random.randn(sample_num, *(th.data_shape[1:]+[1]))
    else:
      x_t = self.add_noise(x_T, self.time_steps-1)


    images = [x_t]
    for t in reversed(range(self.time_steps-delta_t)):
      # Calculate predicted epsilon_theta
      time_emb = self.time_table[[t] * sample_num]
      pred_noise = self.predict(DataSet(data_dict={
        'features': x_t, 'time_emb': time_emb}))

      if clip:
        x_t = self._sample_with_clip(x_t, t, pred_noise)
      else:
        x_t = self._sample_without_clip(x_t, t, pred_noise)

      images.append(x_t)

    # TODO: ...? be very careful
    images = [(x + 1.) / 2. for x in images]

    if return_all_images: return images
    return x_t

  def add_noise(self, x_0, t):
    epsilon = np.random.randn(*x_0.shape)
    indices = t
    t_shape = [-1] + (len(x_0.shape) - 1) * [1]

    sqrt_bar_alpha_t = self.sqrt_bar_alphas[indices]
    sqrt_bar_alpha_t = sqrt_bar_alpha_t.reshape(t_shape)
    sqrt_one_minus_bar_alpha_t = self.sqrt_one_minus_bar_alphas[indices]
    sqrt_one_minus_bar_alpha_t = sqrt_one_minus_bar_alpha_t.reshape(t_shape)

    x_t = (sqrt_bar_alpha_t * x_0 + sqrt_one_minus_bar_alpha_t * epsilon)
    return x_t
