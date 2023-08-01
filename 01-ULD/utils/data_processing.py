import numpy as np

from typing import Tuple

from matplotlib import pyplot as plt

from xomics import MedicalImage
from xomics.data_io.mi_reader import rd_data
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


def get_random_window(arr: np.ndarray, window_size = 128):
  index = np.random.randint(arr.shape[0] - 1)
  arr = np.array([arr[index]])
  a = np.add.reduce(arr, axis=2)
  distr_s = normalize(np.add.reduce(a, axis=2).reshape((-1)))
  a = np.add.reduce(arr, axis=1)
  distr_h = normalize(np.add.reduce(a, axis=1).reshape((-1)))
  distr_w = normalize(np.add.reduce(a, axis=2).reshape((-1)))
  # print(h,w)
  s = windows_choose(distr_s, window_size)
  h = windows_choose(distr_h, window_size)
  w = windows_choose(distr_w, window_size)

  return index, s, h, w


def gen_windows(arr1: np.ndarray, arr2: np.ndarray, batch_size, windows_size = 128):
  features = []
  targets = []
  for _ in range(batch_size):
    index, s, h, w = get_random_window(arr1, windows_size)
    features.append(arr1[index:index+1, s:s+128, h:h+128, w:w+128, :])
    targets.append(arr2[index:index+1, s:s+128, h:h+128, w:w+128, :])
  features = np.concatenate(features)
  targets = np.concatenate(targets)

  return features, targets




if __name__ == '__main__':
  a = rd_data('D:/xai-omics/data/01-ULD/', ['Subject_1-6'], '1-10 dose', 2)
  img1, img2 = gen_windows(a, a, 1)
  print(img1.shape, img2.shape)