from tes.tes_set import TESSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from xomics.objects import MedicalImage
from tqdm import tqdm
from tframe import DataSet

import os
import numpy as np
import random



class TESAgent(DataAgent):

  TFD_FILE_NAME = 'tes.tfd'

  @classmethod
  def load(cls):
    from tes_core import th
    mi_set = cls.load_as_tframe_data(th.data_dir)

    mi_file_test = mi_set.data_dict['mi_file_test'].tolist()
    mi_file_train = mi_set.data_dict['mi_file_train'].tolist()

    mi_list_test = []
    for f in tqdm(mi_file_test, desc='Loading mi files for Test'):
      mi: MedicalImage = MedicalImage.load(f)
      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct'], 'min_max')
      mi.images['ct'] = mi.images['ct'] * 2 - 1
      mi.crop([mi.shape[0], th.xy_size, th.xy_size], random_crop=False,
              basis=list(mi.labels.keys()))
      mi_list_test.append(mi)

    mi_list_train = []
    for f in tqdm(mi_file_train, desc='Loading mi files for Train'):
      mi: MedicalImage = MedicalImage.load(f)
      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct'], 'min_max')
      mi.images['ct'] = mi.images['ct'] * 2 - 1
      mi.crop([mi.shape[0], th.xy_size, th.xy_size], random_crop=False,
              basis=list(mi.labels.keys()))
      mi_list_train.append(mi)

    mi_list = mi_list_test + mi_list_train

    if len(mi_list) == 1:
      train_list, valid_list, test_list = mi_list, mi_list, mi_list
    else:
      # random.shuffle(mi_list_train)
      parts = ratio_to_realnum([9, 1], len(mi_list_train))
      train_list = mi_list_train[:parts[0]]
      valid_list = mi_list_train[parts[0]:]
      test_list = mi_list_test

    ds_list = [train_list, valid_list, test_list]
    ds_name = ['TrainSet', 'ValidSet', 'TestSet']
    datasets = []
    for dl, dn in zip(ds_list, ds_name):
      features, targets, pids = [], [], []
      for mi in dl:
        image = mi.images['ct']
        label = mi.labels['label-0']
        indices = np.where(label == 1)[0]
        image = image[np.min(indices): np.max(indices) + 1]
        label = label[np.min(indices): np.max(indices) + 1]

        feature = [image[i:i + th.slice_num, :, :]
                   for i in range(image.shape[0] - th.slice_num + 1)]
        target = [label[i:i + th.slice_num, :, :]
                  for i in range(label.shape[0] - th.slice_num + 1)]

        features.append(np.squeeze(feature))
        targets.append(np.squeeze(target))
        pids.append(mi.key)

      features = np.expand_dims(np.concatenate(features, axis=0), axis=-1)
      targets = np.expand_dims(np.concatenate(targets, axis=0), axis=-1)

      datasets.append(TESSet(features=features, targets=targets, name=dn))
    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> TESSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return TESSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = TESSet(data_dict=image_dict, name='TESSet')

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

    print(data_dir)
    image_dict = OrderedDict()

    mi_dir_test = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/02-PET-CT-Y1/mi/es')
    mi_dir_test = os.path.abspath(mi_dir_test)
    file_names = os.listdir(mi_dir_test)
    mi_file_test = [os.path.join(mi_dir_test, file) for file in file_names]

    mi_dir_train = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/02-PET-CT-Y1/Public Datasets/SegTHOR/mi')
    mi_dir_train = os.path.abspath(mi_dir_train)
    file_names = os.listdir(mi_dir_train)
    mi_file_train = [os.path.join(mi_dir_train, file) for file in file_names]

    image_dict['mi_file_test'] = np.array(mi_file_test, dtype=object)
    image_dict['mi_file_train'] = np.array(mi_file_train, dtype=object)

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts



if __name__ == '__main__':
  a = TESAgent()
  train_set, _, _ = a.load()

  print()






