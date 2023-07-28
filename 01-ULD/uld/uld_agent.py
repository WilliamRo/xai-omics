import numpy as np

from tframe.data.base_classes import DataAgent
from uld.uld_set import ULDSet, DataSet

import os

from xomics.data_io.mi_reader import rd_subject


class ULDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    ds: ULDSet = cls.load_as_tframe_data(data_dir)

    return ds.split(-1, validate_size, test_size,
                    names=['Train-Set', 'Val-Set', 'Test-Set'])


  @classmethod
  def load_as_tframe_data(cls, data_dir) -> ULDSet | DataSet:
    from uld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    features, targets = cls.load_as_numpy_arrays(data_dir)

    if th.data_arg.func_name == 'alpha':
      s = th.window_size
      # Each x in xs has a shape of [s, s, s, 1]
      xs = []
      ys = []

      h = 250
      w = 350

      # TODO: fill in xs and ys
      for i in range(0, h - s + 1, 30):
        for j in range(0, w - s + 1, 30):
          xs.append(features[:, 4:, 100 + i:100 + i + s, 50 + j:50 + j + s, :])
          ys.append(targets[:, 4:, 100 + i:100 + i + s, 50 + j:50 + j + s, :])

      xs = np.concatenate(xs, axis=0)
      ys = np.concatenate(ys, axis=0)
      print(xs.shape)
      return DataSet(xs, ys)

    return ULDSet(features, targets)


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    from uld_core import th

    data_root = os.path.join(data_dir, th.data_arg.arg_list[0])

    features = rd_subject(data_root, 'Subject_372-387', patient_num=1,
                          dose="1-4 dose")
    targets = rd_subject(data_root, 'Subject_372-387', patient_num=1)

    return features, targets