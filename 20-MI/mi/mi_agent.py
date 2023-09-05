import random

from mi.mi_set import MISet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tools import data_processing
from xomics.objects import MedicalImage

import os
import numpy as np



class MIAgent(DataAgent):

  TFD_FILE_NAME = 'mi.tfd'

  @classmethod
  def load(cls):
    '''
    input: features and targets have the same shape
    features.shape =
    (num_of_examples, slice_num, length, width, channel)
    -> (70, 68, 512, 512, 1)
    It should be noted that the value of the second dimension of feature's
    shape is not fixed.
    '''
    from mi_core import th
    mi_set = cls.load_as_tframe_data(th.data_dir)

    if len(mi_set) == 1:
      # If there is only 1 example, we make training set, validation set
      # and testing set the same.
      datasets = mi_set, mi_set, mi_set
    else:
      ratio = [int(x) for x in th.ratio_of_dataset.split(':')]
      assert len(ratio) == 3
      parts = [int((r / sum(ratio)) * len(mi_set)) for r in ratio[:-1]]
      parts.append(len(mi_set) - sum(parts))
      assert sum(parts) == len(mi_set)

      datasets = mi_set.split(
        parts[0], parts[1], parts[2],
        names=['TrainSet', 'ValSet', 'TestSet'])
    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> MISet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return MISet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = MISet(data_dict=image_dict, name='MISet')

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
    input: XXXXX.npy
    output: features and targets have the same shape
            features.shape =
            (num_of_examples, slice_num, length, width, channel)
            -> (70, 68, 512, 512, 1)
    It should be noted that the value of the second dimension of feature's
    shape is not fixed.
    '''
    print(data_dir)
    image_dict = OrderedDict()

    features, targets = [], []

    data_key = ['XuJiaqi', 'LiTS', 'PET'][2]

    if data_key == 'XuJiaqi':
      file_path = os.path.join(data_dir, 'npy_data', 'ANON10066.npy')
      load_data = np.load(file_path, allow_pickle=True).tolist()

      for key in load_data:
        features.append(load_data[key]['ct_array'])
        targets.append(load_data[key]['ctv_array'])
      features = np.expand_dims(np.array(features), axis=0)
      features = np.expand_dims(features, axis=-1)
      targets = np.expand_dims(np.array(targets), axis=0)
      targets = np.expand_dims(targets, axis=-1)

      image_dict['features'] = features
      image_dict['targets'] = targets

    elif data_key == 'LiTS':
      features_file = os.path.join(
        data_dir, 'npy_data', 'LiTS_train_features.npy')
      targets_file = os.path.join(
        data_dir, 'npy_data', 'LiTS_train_targets.npy')

      features = data_processing.load_npy_file(features_file)
      targets = data_processing.load_npy_file(targets_file)

      # expand the channel dimension
      features = np.array([np.expand_dims(feature, axis=-1)
                           for feature in features], dtype=object)
      targets = np.array([np.expand_dims(target, axis=-1)
                          for target in targets], dtype=object)

      # targets include '0', '1' and '2'
      for i in range(len(targets)):
        mask = targets[i] != 0
        targets[i][mask] = 1

      image_dict['features'] = features
      image_dict['targets'] = targets

    elif data_key == 'PET':
      mi_dir = '../../data/02-PET-CT-Y1/mi'
      mi_dir = os.path.abspath(mi_dir)
      file_names = os.listdir(mi_dir)
      mi_file_list = [os.path.join(mi_dir, file) for file in file_names]

      image_dict['mi_file_list'] = np.array(mi_file_list, dtype=object)

    return image_dict



if __name__ == '__main__':
  a = MIAgent()
  train_set, _, _ = a.load()

  print()






