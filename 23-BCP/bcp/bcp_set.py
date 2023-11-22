import random
import numpy as np
import os

from tframe import DataSet
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from copy import copy
from tools import image_processing



class BCPSet(DataSet):

  def report(self):
    pass

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    '''

    '''
    from bcp_core import th
    round_len = self.get_round_length(batch_size, training=is_training)
    mi_list = self.data_dict['mi_list'].tolist()

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, self.size - 1)
          mi: MedicalImage = copy(mi_list[num])

          # TODO
          # Data Preprocessing
          mi.normalization(['mr'], 'min_max')
          mi.images['mr'] = mi.images['mr'] * 2 - 1

          # Get feature and target
          feature = mi.images['mr']
          target = mi.labels['label-0']

          # TODO
          # Data Augmentation
          if th.random_flip:
            for axe in range(3):
              if random.choice([True, False]) and axe != 0:
                feature = image_processing.image_flip(feature, axes=axe)
                target = image_processing.image_flip(target, axes=axe)
                break

          if th.random_rotation:
            angle = random.choice([0, 90, 180, 270])
            feature = image_processing.image_rotation(feature, angle)
            target = image_processing.image_rotation(target, angle)

          if th.random_noise:
            feature = image_processing.add_gaussian_noise(feature)

          features.append(feature)
          targets.append(target)

        features = np.expand_dims(features, axis=-1)
        targets = np.expand_dims(targets, axis=-1)

        name = 'batch_train_' + str(i)
        data_batch = DataSet(features=features, targets=targets, name=name)
        yield data_batch
    else:
      number = list(range(self.size))
      number_list = [number[i:i+th.val_batch_size]
                     for i in range(0, len(number), th.val_batch_size)]

      for num in number_list:
        features, targets = [], []
        for i in num:
          mi: MedicalImage = copy(mi_list[i])

          # TODO
          # Data Preprocessing
          mi.normalization(['mr'], 'min_max')
          mi.images['mr'] = mi.images['mr'] * 2 - 1

          # Get feature and target
          feature = mi.images['mr']
          target = mi.labels['label-0']

          features.append(feature)
          targets.append(target)

        features = np.expand_dims(features, axis=-1)
        targets = np.expand_dims(targets, axis=-1)
        name = 'batch_val'
        data_batch = DataSet(features=features, targets=targets, name=name)
        yield data_batch

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()


  def visualize_self(self, example_num):
    pass


  def visulization(self, mi_list):
    pass


  def _check_data(self):
    pass


  def test_model(self, model):
    pass



if __name__ == '__main__':
  pass

