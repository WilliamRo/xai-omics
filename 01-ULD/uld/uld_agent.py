import numpy as np

from tframe.data.base_classes import DataAgent
from uld.uld_set import ULDSet, DataSet

import os



class ULDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    from uld_core import th
    ds: ULDSet = cls.load_as_tframe_data(data_dir)
    if th.exp_name in ['alpha', 'beta', 'gamma']:
      return ds.split(-1, validate_size, test_size,
                      names=['Train-Set', 'Val-Set', 'Test-Set'])
    else:
      return ds.get_subsets(-1, validate_size, test_size,
                            names=['Train-Set', 'Val-Set', 'Test-Set'])

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    from uld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    if th.exp_name in ['alpha', 'beta', 'gamma']:
      features, targets = cls.load_as_numpy_arrays(data_dir)
      return cls.early_exp(th, th.exp_name, features, targets)
    else:
      data_root = os.path.join(data_dir, th.data_kwargs['dataset'])
      if th.classify:
        return ULDSet.load_as_uldset(data_root)
      dose = th.data_kwargs['dose']
      return ULDSet.load_as_uldset(data_root, dose)

  # Keep this function to run alpha beta gamma exp
  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    from uld_core import th
    from xomics.data_io.npy_reader import NpyReader
    data_root = os.path.join(data_dir, th.data_kwargs['dataset'])
    dose = th.data_kwargs['dose']
    subjects = [i for i in range(1, 7)]

    reader = NpyReader(data_root)
    features = reader.load_data(subjects, dose)
    targets = reader.load_data(subjects, "Full")

    return features, targets

  @classmethod
  def early_exp(cls, th, expname, features, targets):
    if expname == 'alpha':
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
    if expname == 'alpha2':
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
    if expname in ['beta', 'gamma']:
      return ULDSet(features=features, targets=targets)
