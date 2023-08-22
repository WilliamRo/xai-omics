import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from xomics.data_io.mi_reader import load_data
from utils.metrics_calc import calc_metric, get_metrics

dose_tags = [
  'Full',
  '1-2',
  '1-4',
  '1-10',
  '1-20',
  '1-50',
  '1-100',
  ]





def output_metrics(arr1, arr2, metrics: list):
  result = get_metrics(arr1, arr2, metrics)
  for metric, value in result.items():
    print(f'{metric}: {value}')

  return result


def hist_draw(arr, bins: str | int = 'auto', axis_range=None, log=False, equal=False):
  arr = arr.reshape(-1)
  if equal:
    arr = exposure.equalize_hist(arr)
  arr = arr[arr > 0.015]
  plt.hist(x=arr, bins=bins, range=axis_range, log=log, density=True)
  plt.show()
  return


def slice_hist_draw(arr: np.ndarray):
  arr = np.add.reduce(arr, axis=1)
  arr = np.add.reduce(arr, axis=1).reshape(-1)

  return arr





if __name__ == '__main__':
  from uld_core import th
  th.use_tanh = False
  th.use_clip = 0.15
  dirpath = '../../data/01-ULD/'
  full = load_data(dirpath, 2, dose_tags[0])
  # print(imgs[0].shape)
  full = full[0, ..., 0]
  # low = dose['1-4 dose'][0, ..., 0]
  # delta = full - low

  metr = ['nrmse', 'SSIM', 'psnr', 'rmse']
  # for i in range(len(dose_tags[1:])):
  #   print(f'{dose_tags[1:][i]} dose:')
  #   img = load_data(dirpath, 1, dose_tags[1:][i])[0, ..., 0]
  #   output_metrics(full, img, metr)
  hist_draw(full, bins=50, axis_range=(0, 0.2))
  # slice_distr = slice_hist_draw(full)
  # hist_draw(slice_distr, bins=60)
