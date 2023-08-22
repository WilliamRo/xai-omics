import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, \
  mean_squared_error




def calc_metric(arr1, arr2, metric='mse'):
  metric = metric.lower()
  if metric == 'mse':
    return mean_squared_error(arr1, arr2)
  elif metric == 'rmse':
    return np.sqrt(mean_squared_error(arr1, arr2))
  elif metric == 'ssim':
    return structural_similarity(arr1, arr2, data_range=1)
  elif metric == 'psnr':
    return peak_signal_noise_ratio(arr1, arr2)
  elif metric == 'nrmse':
    return np.sqrt(np.sum(np.square(arr1-arr2))/np.sum(np.square(arr1)))
  else:
    raise ValueError('Unsupported Metric')


def get_metrics(arr1, arr2, metrics: list):
  result = {}
  for metric in metrics:
    result[metric] = calc_metric(arr1, arr2, metric)

  return result