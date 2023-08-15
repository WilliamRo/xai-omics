import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, \
  mean_squared_error
from xomics.data_io.mi_reader import load_data




dose_tags = [
  'Full',
  '1-2',
  '1-4',
  '1-10',
  '1-20',
  '1-50',
  '1-100',
  ]


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


def output_metrics(arr1, arr2, metrics: list):
  result = {}
  for metric in metrics:
    result[metric] = calc_metric(arr1, arr2, metric)
  for metric, value in result.items():
    print(f'{metric}: {value}')

  return result


def hist_draw(arr, range=None, log=False, equal=False):
  arr = arr.reshape(-1)
  if equal:
    arr = exposure.equalize_hist(arr)
  plt.hist(x=arr, bins=50, range=range, log=log)
  plt.show()
  return




if __name__ == '__main__':
  dirpath = '../../data/01-ULD/'
  full = load_data(dirpath, 1, dose_tags[0])
  # print(imgs[0].shape)
  full = full[0, ..., 0]
  # low = dose['1-4 dose'][0, ..., 0]
  # delta = full - low

  metr = ['nrmse', 'SSIM', 'psnr',]
  for i in range(len(dose_tags[1:])):
    print(f'{dose_tags[1:][i]} dose:')
    img = load_data(dirpath, 1, dose_tags[1:][i])[0, ..., 0]
    output_metrics(full, img, metr)

  # hist_draw(low, range=[-0.01, 0.01], equal=0)

