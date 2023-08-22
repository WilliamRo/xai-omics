import numpy as np

from typing import Tuple

from matplotlib import pyplot as plt

from xomics.data_io.mi_reader import load_data
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon


# self.features/targets.shape = [N, S, H, W, 2]
def get_center(arr: np.ndarray, size):
  start_x = (arr.shape[-2] - size) // 2
  start_y = (arr.shape[-3] - size) // 2

  return arr[:, :, start_x:start_x + size, start_y:start_y + size]


def normalize(arr: np.array):
  norm = np.linalg.norm(arr, ord=1)
  return arr / norm


def windows_choose(distr: np.ndarray, windows_size):
  x = np.linspace(0, distr.shape[0] - 1, distr.shape[0])
  result = np.random.choice(x, p=distr)
  result = result - windows_size / 2

  if result < 0: result = 0
  if result > distr.shape[0] - windows_size:
    result = distr.shape[0] - windows_size

  return int(result)


def get_random_window(arr: np.ndarray, window_size=128, slice_size=16):
  # for Gamma test
  # arr = np.where(arr != 0, 1, arr)
  s = np.random.randint(arr.shape[1] - slice_size + 1)

  index = np.random.randint(arr.shape[0])
  arr = arr[index:index+1]
  arr = np.where(arr != 0, 1, arr)
  # arr_pro = np.add.reduce(arr, axis=2)
  # distr_s = normalize(np.add.reduce(arr_pro, axis=2).reshape((-1)))
  arr = arr[:, s:s+slice_size]
  arr_pro = np.add.reduce(arr, axis=1)
  arr_pro = np.add.reduce(arr_pro, axis=3)
  distr_w = normalize(np.add.reduce(arr_pro, axis=1).reshape((-1)))
  distr_h = normalize(np.add.reduce(arr_pro, axis=2).reshape((-1)))
  # print(h,w)
  # s = windows_choose(distr_s, slice_size)
  h = windows_choose(distr_h, window_size)
  w = windows_choose(distr_w, window_size)

  return index, s, h, w


def gen_windows(arr1: np.ndarray, arr2: np.ndarray, batch_size,
                windows_size=128, slice_size=16):
  features = []
  targets = []
  for _ in range(batch_size):
    index, s, h, w = get_random_window(arr1, windows_size, slice_size)
    features.append(arr1[index:index+1, s:s+slice_size,
                    h:h+windows_size, w:w+windows_size, :])
    targets.append(arr2[index:index+1, s:s+slice_size,
                   h:h+windows_size, w:w+windows_size, :])
  features = np.concatenate(features)
  targets = np.concatenate(targets)

  return features, targets




if __name__ == '__main__':
  a = load_data('D:/projects/xai-omics/data/01-ULD/', 1, "Full")
  img = gen_windows(a, a, 1)
  print(a.shape, img[0].shape)
  pass
  # num = 20
  # win = gen_windows(a, a, num, slice_size=32)
  # print(win[0].shape)
  # di = {}
  # for i in range(num):
  #   di[f'test-{i}'] = win[0][i]
  #   print(di[f'test-{i}'].shape)
  #
  # mi = MedicalImage('test', di)
  #
  # dg = DrGordon([mi])
  # dg.slice_view.set('vmin', auto_refresh=False)
  # dg.slice_view.set('vmax', auto_refresh=False)
  # dg.show()
  # print(get_center(arr, 128).shape)
  # prob = calc_prob(arr, 128)
  # print(prob.shape)
