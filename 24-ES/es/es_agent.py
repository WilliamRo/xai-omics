import copy
import random
import os
import numpy as np
import math

from es.es_set import ESSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon


class ESAgent(DataAgent):

  TFD_FILE_NAME = 'es.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from es_core import th
    mi_set = cls.load_as_tframe_data(th.data_dir)

    mi_list = []
    mi_file_list = mi_set.data_dict['mi_file_list'].tolist()
    # mi_file_list = [mi_file_list[0]]

    for f in tqdm(mi_file_list, desc='Loading mi files'):
      mi: MedicalImage = MedicalImage.load(f)
      mi.window('ct', th.window[0], th.window[1])
      mi.crop(th.crop_size, random_crop=False, basis=list(mi.labels.keys()))
      mi.normalization(['ct'], 'z_score')
      mi_list.append(mi)

    if len(mi_list) == 1:
      train_list, valid_list, test_list = mi_list, mi_list, mi_list
    else:
      train_list = mi_list[:18] + mi_list[20:82]
      valid_list = mi_list[18:20] + mi_list[82:-18]
      test_list = mi_list[-18:]

    mi_list = [train_list, valid_list, test_list]
    ds_name = ['TrainSet', 'ValidSet', 'TestSet']
    dataset = []

    pet_mean = np.mean([mi.images['pet'] for mi in train_list])
    pet_std = np.std([mi.images['pet'] for mi in train_list])

    for mi in train_list + valid_list + test_list:
      mi.images['pet'] = (mi.images['pet'] - pet_mean) / pet_std

    for ds_list, name in zip(mi_list, ds_name):
      features, targets = [], []
      region_masks, lesion_masks = [], []
      classes, crop_sizes = [], []
      pids = []
      for mi in ds_list:
        features.append(
          np.stack([mi.images['ct'], mi.images['pet']], axis=-1))

        targets.append(np.expand_dims(mi.labels['lesion'], axis=-1))

        if 'region' in mi.labels.keys():
          region_masks.append(np.expand_dims(
            mi.labels['region'], axis=-1))
          classes.append([0, 1])
          crop_sizes.append(get_min_crop_size(
            [mi.labels['region'], mi.labels['lesion']]))
        else:
          region_masks.append(np.expand_dims(
            np.ones_like(mi.labels['lesion']), axis=-1))
          classes.append([1, 0])
          crop_sizes.append(get_min_crop_size([mi.labels['lesion']]))

        pids.append(mi.key)

      data_dict = {
        'features': np.array(features),
        'targets': np.array(targets),
        'region_mask': np.array(region_masks),
        'lesion_mask': np.array(targets),
        'class': np.array(classes),
        'crop_size': np.array(crop_sizes),
        'pids': np.array(pids, dtype=object)
      }
      dataset.append(ESSet(data_dict=data_dict, name=name))

    return dataset


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> ESSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return ESSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = ESSet(data_dict=image_dict, name='ESSet')

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
    '''
    print(data_dir)
    image_dict = OrderedDict()
    mi_dir = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/02-PET-CT-Y1/mi/both')

    mi_dir = os.path.abspath(mi_dir)
    file_names = os.listdir(mi_dir)
    mi_file_list = [os.path.join(mi_dir, file) for file in file_names]

    image_dict['mi_file_list'] = np.array(mi_file_list, dtype=object)

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts


def get_min_crop_size(input_datas: list):
  assert all(isinstance(arr, np.ndarray) for arr in input_datas)
  assert all(arr.shape == input_datas[0].shape for arr in input_datas)

  union = np.logical_or.reduce(input_datas)
  indices = np.argwhere(union == 1)
  max_indices = np.max(indices, axis=0)
  min_indices = np.min(indices, axis=0)

  delta_indices = [
    maxi - mini for maxi, mini in zip(max_indices, min_indices)]

  crop_size = [next_power_of_2(d) for d in delta_indices]
  z_size = max(crop_size[0], 32)
  xy_size = max(max(crop_size[1:]), 64)

  return [z_size, xy_size, xy_size]


def next_power_of_2(number):
  if number == 0:
    return 0

  if number and not (number & (number - 1)):
    return number

  return 2 ** math.ceil(math.log2(number))



if __name__ == '__main__':
  from es_core import th
  agent = ESAgent()
  train_set, val_set, test_set = agent.load()

  train_set.visualize_self(100)
  print()





