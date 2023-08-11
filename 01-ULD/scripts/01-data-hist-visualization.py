import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, \
  mean_squared_error
from xomics.data_io.mi_reader import rd_data




data_dir = r'../../data/01-ULD/'
subjects = ['Subject_1-6']
patient_num = 1
dose_tags = [
  'Full_dose',
  # '1-2 dose',
  '1-4 dose',
  # '1-10 dose',
  # '1-20 dose',
  # '1-50 dose',
  # '1-100 dose',
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
  dose = {}
  for dose_tag in dose_tags:
    dose[dose_tag] = rd_data(data_dir, subjects, dose_tag, patient_num)
    # print(dose[dose_tag].shape)
  full = dose['Full_dose'][0, ..., 0]
  low = dose['1-4 dose'][0, ..., 0]
  delta = full - low

  # metr = ['mse', 'rmse', 'SSIM']
  # output_metrics(full, low, metr)

  hist_draw(low, range=[-0.01, 0.01], equal=0)

