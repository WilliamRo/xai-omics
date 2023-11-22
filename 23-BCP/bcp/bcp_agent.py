import copy
import random
import os
import numpy as np

from bcp.bcp_set import BCPSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from copy import deepcopy
from xomics.gui.dr_gordon import DrGordon
from scipy.ndimage import label, generate_binary_structure


class BCPAgent(DataAgent):

  TFD_FILE_NAME = 'bcp.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from bcp_core import th

    bcp_set = cls.load_as_tframe_data(th.data_dir)
    mi_list = bcp_set.data_dict['mi_list'].tolist()

    if len(bcp_set) == 1:
      # If there is only 1 example, we make training set, validation set
      # and testing set the same.
      datasets = bcp_set, bcp_set, bcp_set
    else:
      train_list = mi_list[:-1]
      val_list = mi_list[-1:]
      test_list = deepcopy(mi_list[-1:])

      ds_list = [train_list, val_list, test_list]
      ds_name = ['TrainSet', 'ValidSet', 'TestSet']
      datasets = [
        BCPSet(data_dict={'mi_list': np.array(dl, dtype=object)}, name=dn)
        for dl, dn in zip(ds_list, ds_name)]

    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> BCPSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return BCPSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    # data_set = BCPSet(data_dict=image_dict, name='BCPSet')
    data_set = BCPSet(data_dict=image_dict, name='BCPSet')

    # Show status
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving datas set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    # Wrap and return

    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir) -> OrderedDict:
    '''
      features and targets are exactly the same.
      In this dataset, the shape of pet data is (175, 440, 440), which is
      about 130 MB
    '''
    print(data_dir)
    image_dict = OrderedDict()

    dir = os.path.abspath(
      os.path.join(data_dir, '../../data/05-Brain-MR/mi'))

    filenames = [f for f in os.listdir(dir) if '.mi' in f]

    mi_list = [MedicalImage.load(os.path.join(dir, file))
               for file in tqdm(filenames, desc='Reading mi files')]

    image_dict['mi_list'] = np.array(mi_list, dtype=object)

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts



if __name__ == '__main__':
  from bcp_core import th
  agent = BCPAgent()
  train_set, val_set, test_set = agent.load()
  print()





