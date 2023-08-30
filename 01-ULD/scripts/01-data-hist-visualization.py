import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from xomics.data_io.npy_reader import load_data

dose_tags = [
  'Full',
  '1-2',
  '1-4',
  '1-10',
  '1-20',
  '1-50',
  '1-100',
  ]


def hist_draw(arr, bins: str | int = 'auto', axis_range=None, log=False, equal=False):
  arr = arr.reshape(-1)
  if equal:
    arr = exposure.equalize_hist(arr)
  arr = arr[arr > 0]

  plt.hist(x=arr, bins=bins, range=axis_range, log=log)
  plt.xlabel("Normalized Intensity")
  plt.ylabel("Number")
  plt.title("PET Image Histogram")

  plt.show()
  return


def draw_outs(arr, gsize, pos, range, bins, rowspan=1, colspan=1):
  ax1 = plt.subplot2grid(gsize, pos, rowspan=rowspan, colspan=colspan)
  ax1.hist(x=arr, bins=bins, range=range)
  return ax1


def mapping(arr: np.ndarray):
  scope_list = []
  func_list = []
  return np.piecewise(arr, scope_list, func_list)




if __name__ == '__main__':
  dirpath = '../../data/01-ULD/'
  full = load_data(dirpath, 6, dose_tags[0])
  # print(imgs[0].shape)
  full = full[0, ..., 0]
  print(full.shape)
  # full[full < 0.00005] = 0
  hist_draw(full, bins=50, axis_range=(0, 1))

  # slice_distr = slice_hist_draw(full)
