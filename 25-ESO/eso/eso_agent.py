import copy
import random
import os
import numpy as np
import math

from eso.eso_set import ESOSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from itertools import chain
from xomics.gui.dr_gordon import DrGordon


class ESOAgent(DataAgent):

  TFD_FILE_NAME = 'eso.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from eso_core import th
    mi_set = cls.load_as_tframe_data(th.data_dir)

    mi_list = []
    mi_file_list = mi_set.data_dict['mi_file_list'].tolist()
    # mi_file_list = mi_file_list[:33]

    for f in tqdm(mi_file_list, desc='Loading mi files'):
      mi: MedicalImage = MedicalImage.load(f)
      mi.window('ct', th.window[0], th.window[1])

      mi.normalization(['ct'], 'min_max')
      mi.images['ct'] = mi.images['ct'] * 2 - 1
      mi.images['ct'] = mi.images['ct'].astype(np.float32)

      bottom, top = mi.crop(th.crop_size, random_crop=False, basis=['lesion'])
      mi_list.append(mi)
      mi.put_into_pocket('crop_info', [bottom, top])

    if len(mi_list) <= 10:
      train_list, valid_list, test_list = mi_list, mi_list, mi_list
    else:
      # train_list = mi_list[:18] + mi_list[20:82]
      # valid_list = mi_list[18:20] + mi_list[82:-18]
      # test_list = mi_list[-18:]
      if th.cross_validation:
        train_val_list, test_list = cv_split(
          mi_list, th.k_fold, th.val_fold_index)
        train_list, valid_list = split_list(train_val_list, '7:1')
      else:
        train_list, valid_list, test_list = split_list(
          mi_list, th.ratio_of_dataset)

    mi_list = [train_list, valid_list, test_list]
    ds_name = ['TrainSet', 'ValidSet', 'TestSet']
    dataset = []

    pet_mean = np.mean([mi.images['pet'] for mi in train_list])
    pet_std = np.std([mi.images['pet'] for mi in train_list])

    for mi in train_list + valid_list + test_list:
      mi.images['pet'] = (mi.images['pet'] - pet_mean) / pet_std

    for ds_list, name in zip(mi_list, ds_name):
      features, targets = [], []
      pids, crop_info = [], []
      for mi in ds_list:
        features.append(np.stack([mi.images['ct'], mi.images['pet'],
                                  mi.labels['region']], axis=-1))

        targets.append(np.expand_dims(mi.labels['lesion'], axis=-1))

        pids.append(mi.key)
        crop_info.append(mi.get_from_pocket('crop_info'))

      data_dict = {
        'features': np.array(features),
        'targets': np.array(targets),
        'pids': np.array(pids, dtype=object),
        'crop_info': np.array(crop_info)
      }
      dataset.append(ESOSet(data_dict=data_dict, name=name))

    return dataset


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> ESOSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return ESOSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = ESOSet(data_dict=image_dict, name='ESOSet')

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
      'data/02-PET-CT-Y1/mi/both_allmask')

    mi_dir = os.path.abspath(mi_dir)
    file_names = os.listdir(mi_dir)

    file_names = sorted(file_names, key=sorting_key, reverse=False)
    print(file_names)

    mi_file_list = [os.path.join(mi_dir, file) for file in file_names]

    image_dict['mi_file_list'] = np.array(mi_file_list, dtype=object)

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts


def split_list(input_list, ratio):
  ratio = [int(r) for r in ratio.split(':')]
  n = sum(ratio)
  folds, results = [], []

  avg_length, remainder = len(input_list) // n, len(input_list) % n

  start = 0
  for i in range(n):
    end = start + avg_length + 1 if i < remainder else start + avg_length
    folds.append(input_list[start:end])
    start = end

  for r in ratio:
    results.append(list(chain(*folds[:r])))
    folds = folds[r:]

  return results


def cv_split(input_list, n_fold, val_fold_num):
  folds = []

  avg_length, remainder = len(input_list) // n_fold, len(input_list) % n_fold
  start = 0
  for i in range(n_fold):
    end = start + avg_length + 1 if i < remainder else start + avg_length
    folds.append(input_list[start:end])
    start = end

  val = folds[val_fold_num]
  train = list(chain(*[folds[i] for i in range(n_fold) if i != val_fold_num]))

  return train, val


def sorting_key(item):
  return int(item.split('_')[0])



if __name__ == '__main__':
  from eso_core import th
  agent = ESOAgent()
  train_set, val_set, test_set = agent.load()

  train_set.visualize_self(100)
  print()





