from tframe.data.base_classes import DataAgent
from rld.rld_set import RLDSet
from xomics.objects.general_mi import GeneralMI

import os
import numpy as np



class RLDAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, validate_size, test_size):
    ds: RLDSet = cls.load_as_tframe_data(data_dir)
    test_pid = ['YHP00012417', 'YHP00010651', 'YHP00012231',
                'YHP00012016', 'YHP00011840']
    train, test = ds.subset(test_pid, 'Test-Set')
    train, valid = train.split(-1, validate_size,
                               names=['Train-Set', 'Val-Set'])
    # train.mi_data.get_stat()
    # valid.mi_data.get_stat()
    # test.mi_data.get_stat()
    return  train, valid, test

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

    img_keys = [th.data_set[0], 'CT_seg']
    if not th.noCT:
      img_keys += ['CT']

    img_type = {
      'CT': ['CT'],
      'PET': ['30G', '20S', '40S', '60G', '120S', '240G', '240S'],
      'MASK': ['CT_seg'],
      'STD': th.data_set[:1],
    }

    mi = GeneralMI(img_dict, image_keys=img_keys, process_param=th.process_param,
                   label_keys=[th.data_set[1]], pid=pid, img_type=img_type)
    return RLDSet(mi_data=mi, buffer_size=th.buffer_size)

