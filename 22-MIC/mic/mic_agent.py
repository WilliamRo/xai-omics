import random

from mic.mic_set import MICSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from tools import data_processing
from tqdm import tqdm
from xomics import MedicalImage

import os
import numpy as np



class MICAgent(DataAgent):

  TFD_FILE_NAME = 'mic.tfd'

  @classmethod
  def load(cls):
    '''

    '''
    from mic_core import th

    mic_set = cls.load_as_tframe_data(th.data_dir)

    ratio = [int(x) for x in th.ratio_of_dataset.split(':')]
    assert len(ratio) == 3

    if not th.cross_validation:
      datasets = mic_set.split(
        ratio[0], ratio[1], ratio[2],
        names=['TrainSet', 'ValSet', 'TestSet'], over_classes=True)
    else:
      # TODO
      train_val_set, test_set = mic_set.split(
        ratio[0] + ratio[1], ratio[2],
        names=['train_val_set', 'TestSet'], over_classes=True)

      BA_indice = train_val_set.groups[0]
      MIA_indice = train_val_set.groups[1]

      BA_partition_size = len(BA_indice) // th.num_fold
      MIA_partition_size = len(MIA_indice) // th.num_fold

      val_num = random.randint(0, th.num_fold - 1)

      start_BA = val_num * BA_partition_size
      end_BA = ((val_num + 1) * BA_partition_size
                if val_num != th.num_fold - 1 else len(BA_indice))
      start_MIA = val_num * MIA_partition_size
      end_MIA = ((val_num + 1) * MIA_partition_size
                 if val_num != th.num_fold - 1 else len(MIA_indice))

      val_indice = (BA_indice[start_BA : end_BA] +
                    MIA_indice[start_MIA : end_MIA])
      train_indice = [x for x in (BA_indice + MIA_indice)
                      if x not in val_indice]

      train_set = train_val_set[train_indice]
      train_set.name = 'TrainSet'
      val_set = train_val_set[val_indice]
      val_set.name = 'ValSet'

      datasets = train_set, val_set, test_set

    return datasets


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> MICSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return MICSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = MICSet(data_dict=image_dict, name='MICSet', NUM_CLASSES=2,
                      classes=['BA', 'MIA'])


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

    mi_dir =  os.path.abspath(r'../../data/BA & MIA/mi')
    BA_dir = os.path.join(mi_dir, 'BA')
    MIA_dir = os.path.join(mi_dir, 'MIA')
    BA_file_names = os.listdir(BA_dir)
    MIA_file_names = os.listdir(MIA_dir)

    mi_file_list_BA = [
      os.path.join(BA_dir, file) for file in BA_file_names]
    mi_file_list_MIA = [
      os.path.join(MIA_dir, file) for file in MIA_file_names]

    mi_file_list = mi_file_list_BA + mi_file_list_MIA
    labels = ([np.array([1, 0]) for _ in BA_file_names] +
              [np.array([0, 1]) for _ in MIA_file_names])

    image_dict['features'] = np.array(mi_file_list, dtype=object)
    image_dict['targets'] = np.array(labels)

    return image_dict


def find_nonzero_bounds(array):
  indices = np.nonzero(array)
  min_z, max_z = np.min(indices[0]), np.max(indices[0])
  min_y, max_y = np.min(indices[1]), np.max(indices[1])
  min_x, max_x = np.min(indices[2]), np.max(indices[2])
  return min_z, max_z, min_y, max_y, min_x, max_x


def extract_minimal_cuboid(array):
  min_z, max_z, min_y, max_y, min_x, max_x = find_nonzero_bounds(array)
  minimal_cuboid = array[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
  return minimal_cuboid


def find_max_dimensions(features):
  max_shape = [0, 0, 0]

  for feature in features:
    shape = feature.shape
    for i in range(3):
      max_shape[i] = max(max_shape[i], shape[i])

  return max_shape


def resize_features(features):
  resized_features = []
  max_shape = find_max_dimensions(features)
  max_shape = [2**find_n_for_range(shape) for shape in max_shape]

  for feature in features:
    new_feature = np.zeros(max_shape, dtype=feature.dtype)
    new_feature[:feature.shape[0], :feature.shape[1],
    :feature.shape[2]] = feature
    resized_features.append(new_feature)

  return resized_features


def find_n_for_range(x):
  n = 1
  while True:
    lower_bound = 2 ** (n - 1)
    upper_bound = 2 ** n
    if lower_bound < x < upper_bound:
      return n
    n += 1



if __name__ == '__main__':
  a = MICAgent()
  train_set, val_set, test_set = a.load()

  train_set.visualize_self(10)



