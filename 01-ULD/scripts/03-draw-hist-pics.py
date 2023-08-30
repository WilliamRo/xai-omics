from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from skimage.exposure.exposure import equalize_hist

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
    arr = equalize_hist(arr)
  arr = arr[arr > 0]

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


if __name__ == '__main__':
  dirpath = '../../data/01-ULD/'
  full = load_data(dirpath, 6, dose_tags[0])
  full = full[0, ..., 0]
  print(full.shape)
  # full[full < 0.00005] = 0
  hist_draw(full, bins=50, axis_range=(0, 1))

