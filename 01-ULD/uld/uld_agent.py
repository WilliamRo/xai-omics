import numpy as np

from tframe.data.base_classes import DataAgent
from uld.uld_set import ULDSet, DataSet

import os

from xomics.data_io.mi_reader import rd_data


class ULDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    from uld_core import th
    ds: ULDSet = cls.load_as_tframe_data(data_dir)
    # if not th.train:
    #   return ds,ds,ds
    return ds.split(-1, validate_size, test_size,
                    names=['Train-Set', 'Val-Set', 'Test-Set'])

  @classmethod
  def load_as_tframe_data(cls, data_dir) -> ULDSet | DataSet:
    from uld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    features, targets = cls.load_as_numpy_arrays(data_dir)
    # if not th.train:
    #   return DataSet(features[:, :, 12:-12, 12:-12], targets[:, :, 12:-12, 12:-12])
    if th.data_arg.func_name == 'alpha':
      s = th.window_size
      # Each x in xs has a shape of [s, s, s, 1]
      xs = []
      ys = []
      h = 150 + 150
      w = 100 + 250

      # fill in xs and ys
      for i in range(150, h - s + 1, 30):
        for j in range(100, w - s + 1, 30):
          xs.append(features[:, :, i:i + s, j:j + s, :])
          ys.append(targets[:, :, i:i + s, j:j + s, :])

      xs = np.concatenate(xs, axis=0)
      ys = np.concatenate(ys, axis=0)
      print(xs.shape)
      return DataSet(xs, ys)
    if th.data_arg.func_name == 'alpha2':
      xs = []
      ys = []

      for i in range(features.shape[1] - 7):
        xs.append(features[:, :, i:i + 8, 12:-12, 12:-12])
        ys.append(targets[:, :, i:i + 8, 12:-12, 12:-12])
      # print(xs[0].shape)
      xs = np.concatenate(xs, axis=0)
      ys = np.concatenate(ys, axis=0)
      # print(xs.shape)
      return DataSet(xs, ys)
    # before beta plan may not be runnable
    if th.data_arg.func_name in ['beta', 'gamma']:
      return ULDSet(features, targets)

    return ULDSet(features, targets)

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    from uld_core import th

    data_root = os.path.join(data_dir, th.data_arg.arg_list[0])
    subjects = ['Subject_1-6']#, 'Subject_7-12', 'Subject_13-18']
                # 'Subject_19-24', 'Subject_25-30']

    features = rd_data(data_root, subjects, patient_num=6,
                       dose="1-4 dose", hist_equal=True)
    targets = rd_data(data_root, subjects, patient_num=6, hist_equal=True)

    return features, targets
