import random
import numpy as np
import os

from tframe import DataSet
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from copy import copy



class BCPSet(DataSet):

  def report(self):
    pass

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    '''

    '''
    from bcp_core import th
    round_len = self.get_round_length(batch_size, training=is_training)

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, self.size - 1)
          feature = np.squeeze(copy(self.features[num]))
          target = np.squeeze(copy(self.targets[num]))

          # Data Augmentation
          # 1. Flip images
          if th.random_flip and random.choice([True, False]):
            axe = random.choice([0, 1, 2])
            feature = image_flip(feature, axes=axe)
            target = image_flip(target, axes=axe)

          # 2. Rotate image
          if th.random_rotation:
            angle = random.choice([0, 90, 180, 270])
            feature = image_rotation(feature, angle)
            target = image_rotation(target, angle)

          # 3. Add gaussian noise
          if th.random_noise:
            feature = add_gaussian_noise(feature)

          features.append(feature)
          targets.append(target)

        # expand the channel dimension
        features = np.expand_dims(np.array(features), axis=-1)
        targets = np.expand_dims(np.array(targets), axis=-1)

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
          features.append(self.features[i])
          targets.append(self.targets[i])
        name = 'batch_val'
        data_batch = DataSet(
          features=np.array(features), targets=np.array(targets), name=name)
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


def add_gaussian_noise(image: np.ndarray, mean=0, std=1):
  assert len(image.shape) in [2, 3]

  noise = np.random.normal(mean, std, image.shape)
  return image + noise


def image_rotation(image, angle):
  assert len(image.shape) in [2, 3]
  assert angle in [0, 90, 180, 270]
  for _ in range(3):
    if angle <= 0: break
    image = [np.rot90(img) for img in image]
    angle = angle - 90

  return np.array(image)


def image_flip(image, axes):
  assert len(image.shape) in [2, 3]
  return np.flip(image, axis=axes)



if __name__ == '__main__':
  pass

