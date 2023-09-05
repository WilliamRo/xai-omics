import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, \
  mean_squared_error




def calc_metric(arr1, arr2, metric='mse', data_range=1.0):
  """
  :param arr1: target
  :param arr2: feature
  :param metric:
  :param data_range:
  :return:
  """
  metric = metric.lower()
  if metric == 'mse':
    return mean_squared_error(arr1, arr2)
  elif metric == 'rmse':
    return np.sqrt(mean_squared_error(arr1, arr2))
  elif metric == 'ssim':
    # assert len(arr1.shape) == 3
    return structural_similarity(arr1, arr2, data_range=data_range)
  elif metric == 'psnr':
    return peak_signal_noise_ratio(arr1, arr2, data_range=data_range)
  elif metric == 'nrmse':
    return np.sqrt(np.sum(np.square(arr1-arr2))/np.sum(np.square(arr1)))
  elif metric == 'pw_rmse':
    return np.sqrt(np.mean(np.square(arr1 - arr2) / (np.square(arr1) + 0.001)))
  else:
    raise ValueError('Unsupported Metric')


def get_metrics(arr1, arr2, metrics: list, data_range=1.0):
  """
  :param arr1: truth
  :param arr2: predict
  :param metrics:
  :param data_range:
  :return:
  """
  result = {}
  for metric in metrics:
    result[metric] = calc_metric(arr1, arr2, metric, data_range)

  return result