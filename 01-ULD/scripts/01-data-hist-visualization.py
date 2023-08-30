import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from skimage import exposure
from xomics.data_io.npy_reader import load_data
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
  arr = arr[arr > 0]
  # fig, ax = plt.subplots(6, 3, figsize=(16, 9),
  #                        gridspec_kw={
  #                          'width_radios': [],
  #                          'height_radios': []
  #                        })
  fig = plt.figure(figsize=(16, 9))
  gs = fig.add_gridspec(4, 3)
  ax = fig.add_subplot(gs[0, :])
  ax.hist(x=arr, bins=bins, range=axis_range, log=log)
  ax.set_xlabel("Normalized Intensity")
  ax.set_ylabel("Number")
  ax.set_title("PET Image Histogram")

  ax1 = fig.add_subplot(gs[2, 0])
  ax1.hist(x=arr, bins=bins, range=(0.0005, 0.025))
  mark_inset(ax, ax1, loc1=1, loc2=2, fc="none", ec="k", lw=1)

  ax2 = fig.add_subplot(gs[2, 1])
  ax2.hist(x=arr, bins=bins, range=(0.07, 0.3))
  mark_inset(ax, ax2, loc1=1, loc2=2, fc="none", ec="k", lw=1)

  ax3 = fig.add_subplot(gs[2, 2])
  ax3.hist(x=arr, bins=bins, range=(0.4, 1.0))
  mark_inset(ax, ax3, loc1=1, loc2=2, fc="none", ec="k", lw=1)

  # axins = draw_ins(ax, arr, (0.05, 0.35, 0.4, 0.4), (0.05, 0.1), bins)
  # draw_ins(ax, arr, (0.48, 0.3, 0.25, 0.25), (0.35, 0.5), bins)
  # draw_ins(ax, arr, (0.78, 0.3, 0.2, 0.2), (0.75, 0.9), bins)
  # draw_ins(axins, arr, (0.3, 0.3, 0.4, 0.4), (0.06, 0.075), bins)
  plt.show()
  # plt.savefig("out.png", dpi=300)
  return


def draw_outs(arr, gsize, pos, range, bins, rowspan=1, colspan=1):
  ax1 = plt.subplot2grid(gsize, pos, rowspan=rowspan, colspan=colspan)
  ax1.hist(x=arr, bins=bins, range=range)
  return ax1


def draw_ins(ax, arr, pos, range, bins):
  axins = ax.inset_axes(pos)
  axins.hist(x=arr, bins=bins, range=range)
  mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)
  return axins


def slice_hist_draw(arr: np.ndarray):
  arr = np.add.reduce(arr, axis=1)
  arr = np.add.reduce(arr, axis=1).reshape(-1)

  return arr


def mapping(arr: np.ndarray):
  scope_list = []
  func_list = []
  return np.piecewise(arr, scope_list, func_list)





if __name__ == '__main__':
  from uld_core import th
  th.use_tanh = 0
  th.norm_by_feature = True
  # th.use_clip = 0
  dirpath = '../../data/01-ULD/'
  full = load_data(dirpath, 6, dose_tags[0])
  # print(imgs[0].shape)
  full = full[0, ..., 0]
  print(full.shape)
  # low = dose['1-4 dose'][0, ..., 0]
  # delta = full - low
  # full[full < 0.00005] = 0
  metr = ['nrmse', 'SSIM', 'psnr', 'rmse']
  # for i in range(len(dose_tags[1:])):
  #   print(f'{dose_tags[1:][i]} dose:')
  #   img = load_data(dirpath, 6, dose_tags[1:][i])[0, ..., 0]
  #   output_metrics(full, img, metr)
  # full_p = mapping(full)
  hist_draw(full, bins=50, axis_range=(0, 1))

  # slice_distr = slice_hist_draw(full)
  # hist_draw(slice_distr, bins=60)
