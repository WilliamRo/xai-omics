from tframe.data.base_classes import DataAgent
from rld.rld_set import RLDSet
from xomics.data_io.reader.general_mi import GeneralMI

import os
import numpy as np



class RLDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    from rld_core import th
    ds: RLDSet = cls.load_as_tframe_data(data_dir)
    return ds.split(-1, validate_size, test_size,
                    names=['Train-Set', 'Val-Set', 'Test-Set'])

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    from rld_core import th

    # features/targets.shape = [N, S, H, W, 1]
    data = np.genfromtxt(os.path.join(data_dir, th.data_kwargs['dataset'], 'rld_data.csv'),
                         delimiter=',', dtype=str)
    types = data[0][1:]
    pid = data[1:, 0]
    path_array = data[1:, 1:]

    img_dict = {}
    for i, type_name in enumerate(types):
      img_path = path_array[:, i]
      img_dict[type_name] = {'path': img_path}

    mi = GeneralMI(img_dict, image_key=th.data_set[0],
                   label_key=th.data_set[1], pid=pid)
    return RLDSet(mi_data=mi, buffer_size=th.buffer_size)

