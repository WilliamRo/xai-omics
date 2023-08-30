import os
import time

import numpy as np

from utils.data_processing import get_random_window
from xomics import MedicalImage
from xomics.data_io.npy_reader import load_data
from xomics.gui.dr_gordon import DrGordon


def record(windows_size, slice_size):
  arr = np.zeros((1, 608, 440, 440, 1))
  count = 0
  for _ in range(num):
    if count % 100 == 0:
      print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}: {count}/{num}")
    index, s, h, w = get_random_window(a, windows_size, slice_size)
    arr[index:index+1, s:s + slice_size, h:h + windows_size, w:w + windows_size] += 1
    count += 1
  arr /= num

  return arr




if __name__ == '__main__':
  num = 100000
  win_size = s_size = 128
  dir_list = os.listdir("./tmp")

  if f"test{num}.npy" not in dir_list:
    a = load_data('../../data/01-ULD/', [1, 3, 5, 8, 10, 12], "Full")
    result = record(win_size, s_size)
    np.save(f"test{num}.npy", result)
  else:
    result = np.load(f"./tmp/test{num}.npy")

  mi = MedicalImage("test", {f"{num}": result[0]})
  dg = DrGordon([mi])
  dg.slice_view.set('vmin', auto_refresh=False)
  dg.slice_view.set('vmax', auto_refresh=False)
  dg.show()
