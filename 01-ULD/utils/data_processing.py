import numpy as np

from typing import Tuple


# self.features/targets.shape = [N, S, H, W, 1]
def get_center(arr: np.ndarray, size):
  start_x = (arr.shape[-2] - size) // 2
  start_y = (arr.shape[-3] - size) // 2

  return arr[:, :, start_x:start_x + size, start_y:start_y + size]


def calc_prob(arr: np.ndarray, size: Tuple[int, int]):
  windowed_arr = sliding_window(arr, size)
  mean_arr = np.mean(windowed_arr, axis=(3, 4))
  prob_arr = mean_arr
  return prob_arr


def sliding_window(arr: np.ndarray, window_size: Tuple[int, int]):
  shape = (arr.shape[0], arr.shape[1], arr.shape[2] - window_size[0] + 1,
           arr.shape[3] - window_size[1] + 1, arr.shape[4]) + \
          (arr.shape[0], arr.shape[1]) + window_size + (1,)
  print(arr.strides)
  strides = arr.strides[:2] + (arr.strides[1] * 2, arr.strides[2] * 2) + \
            arr.strides[-1:]
  return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


if __name__ == '__main__':
  a = np.random.random([20, 2, 5, 5, 1])
  print(calc_prob(a, (4, 4)))
  # # print(get_center(arr, 128).shape)
  # prob = calc_prob(arr, 128)
  # print(prob.shape)
