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
    th.ratio_of_dataset = '6:1:1'

    bcp_set = cls.load_as_tframe_data(th.data_dir)
    mi_list = bcp_set.data_dict['mi_list'].tolist()

    # Pre-processing
    for mi in tqdm(mi_list, desc='Pre processing'):
      assert isinstance(mi, MedicalImage)
      mi.normalization(['mr'], 'min_max')
      mi.images['mr'] = mi.images['mr'] * 2 - 1

      mi.crop([mi.shape[0], th.xy_size, th.xy_size], False, [])

    if len(bcp_set) == 1:
      # If there is only 1 example, we make training set, validation set
      # and testing set the same.
      datasets = bcp_set, bcp_set, bcp_set
    else:
      ratio = th.ratio_of_dataset.split(':')
      ratio = [int(a) for a in ratio]
      part = ratio_to_realnum(ratio, len(mi_list))

      train_list = mi_list[:part[0]]
      val_list = mi_list[part[0]:part[0] + part[1]]
      test_list = mi_list[-part[2]:]

      ds_list = [train_list, val_list, test_list]
      ds_name = ['TrainSet', 'ValidSet', 'TestSet']
      # datasets = [
      #   BCPSet(data_dict={'mi_list': np.array(dl, dtype=object)}, name=dn)
      #   for dl, dn in zip(ds_list, ds_name)]
      datasets = []
      for dl, dn in zip(ds_list, ds_name):
        features, targets = [], []
        for mi in dl:
          image = mi.images['mr']
          label = mi.labels['label-0']

          # TODO
          indices = np.where(label == 1)[0]
          image = image[np.min(indices): np.max(indices) + 1]
          label = label[np.min(indices): np.max(indices) + 1]
          # TODO

          feature = [image[i:i + th.slice_num, :, :]
                     for i in range(image.shape[0] - th.slice_num + 1)]
          target = [label[i:i + th.slice_num, :, :]
                    for i in range(label.shape[0] - th.slice_num + 1)]

          features.append(np.squeeze(feature))
          targets.append(np.squeeze(target))

        features = np.expand_dims(np.concatenate(features, axis=0), axis=-1)
        targets = np.expand_dims(np.concatenate(targets, axis=0), axis=-1)

        datasets.append(BCPSet(features=features, targets=targets, name=dn))

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

    mi_dir = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/05-Brain-MR/mi/NFBS')
    mi_dir = os.path.abspath(mi_dir)

    filenames = [f for f in os.listdir(mi_dir) if '.mi' in f]

    mi_list = [MedicalImage.load(os.path.join(mi_dir, file))
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





