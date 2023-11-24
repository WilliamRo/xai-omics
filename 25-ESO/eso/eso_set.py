import random
import time

import numpy as np
import math
import os

from tframe import DataSet
from roma import console
from tools import image_processing
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from copy import copy



class ESOSet(DataSet):

  def report(self):
    pass


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    from eso_core import th

    round_len = self.get_round_length(batch_size, training=is_training)
    crop_size = [128, 256, 256]

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, self.size - 1)
          feature = np.squeeze(self.features[num])
          target = np.squeeze(self.targets[num])

          ct = feature[:, :, :, 0]
          pet = feature[:, :, :, 1]
          region_mask = feature[:, :, :, 2]


          # Data Augmentation
          cropped_data = image_processing.crop_3d(
            [ct, pet, target, region_mask], crop_size,
            th.random_crop, [target])

          ct, pet, target, region_mask = (cropped_data[0], cropped_data[1],
                                          cropped_data[2], cropped_data[3])

          if th.random_flip:
            for axe in range(3):
              if random.choice([True, False]) and axe != 0:
                ct = image_processing.image_flip(ct, axes=axe)
                pet = image_processing.image_flip(pet, axes=axe)
                target = image_processing.image_flip(target, axes=axe)
                region_mask = image_processing.image_flip(region_mask,
                                                         axes=axe)
                break

          if th.random_rotation:
            angle = random.choice([0, 90, 180, 270])
            ct = image_processing.image_rotation(ct, angle)
            pet = image_processing.image_rotation(pet, angle)
            target = image_processing.image_rotation(target, angle)
            region_mask = image_processing.image_rotation(region_mask, angle)

          if th.random_noise:
            ct = image_processing.add_gaussian_noise(ct)
            pet = image_processing.add_gaussian_noise(pet)

          # Features and Targets
          features.append(
            np.stack([ct, pet, region_mask], axis=-1))
          targets.append(np.expand_dims(target, axis=-1))

        name = 'batch_train_' + str(i)
        data_batch = DataSet(
          features=np.array(features), targets=np.array(targets), name=name)
        yield data_batch
    else:
      number = list(range(self.size))
      number_list = [number[i:i+th.val_batch_size]
                     for i in range(0, len(number), th.val_batch_size)]

      for num in number_list:
        features, targets = [], []

        for i in num:
          feature = np.squeeze(self.features[i])
          target = np.squeeze(self.targets[i])

          ct = feature[:, :, :, 0]
          pet = feature[:, :, :, 1]
          region_mask = feature[:, :, :, 2]

          # Data Augmentation
          cropped_data = image_processing.crop_3d(
            [ct, pet, target, region_mask], crop_size, False, [target])

          ct, pet, target, region_mask = (cropped_data[0], cropped_data[1],
                                          cropped_data[2], cropped_data[3])

          # Features and Targets
          features.append(np.stack([ct, pet, region_mask], axis=-1))
          targets.append(np.expand_dims(target, axis=-1))

        name = 'batch_val'
        data_batch = DataSet(
          features=np.array(features), targets=np.array(targets), name=name)
        yield data_batch

    if is_training: self._clear_dynamic_round_len()


  def _check_data(self):
    pass


  def test_model(self, model):
    from eso_core import th
    from tframe import Predictor
    assert isinstance(model, Predictor)

    print(f'Test Model in {self.name}')
    crop_size = [128, 256, 256]

    pids = self.data_dict['pids'].tolist()
    features = np.squeeze(self.features)
    targets = np.squeeze(self.targets)

    assert len(features) == len(pids)

    predictions = model.predict(self)
    predictions = np.squeeze(predictions)
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    assert len(targets) == len(predictions)

    mi_save_dir = r'E:\xai-omics\data\02-PET-CT-Y1\results\25-ESO\mi'
    model_name = r'1124_unet(4-5-4-1-relu-mp)_Sc_1'
    mi_save_dir = os.path.join(mi_save_dir, model_name)
    if not os.path.exists(mi_save_dir):
      os.mkdir(mi_save_dir)

    mi_list, acc_list = [], []
    for f, t, k, p in zip(features, targets, pids, predictions):
      ct = f[..., 0]
      pet = f[..., 1]
      region_mask = f[..., 2]

      cropped_data = image_processing.crop_3d(
        [ct, pet, t, region_mask], crop_size, False, [t])

      ct, pet, t, region_mask = (
        cropped_data[0], cropped_data[1], cropped_data[2], cropped_data[3])

      images = {'ct': ct, 'pet': pet}
      labels = {'lesion_gt': t, 'region': region_mask, 'lesion_pred': p}
      acc = self.dice_accuarcy(t, p)
      key = k + f' --- Dice Acc: {round(acc, 2)}'
      print(key)
      mi: MedicalImage = MedicalImage(images=images, labels=labels, key=key)

      mi.save(os.path.join(mi_save_dir, k + '.mi'))
      mi_list.append(mi)
      acc_list.append(acc)

    print(f'Dice Accuracy in Average: {round(np.mean(acc_list), 2)}')

    dg = DrGordon(mi_list)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.show()


  def dice_accuarcy(self, ground_truth, prediction):
    assert ground_truth.shape == prediction.shape
    smooth = 1.0

    intersection = np.sum(ground_truth * prediction)
    acc = ((2.0 * intersection + smooth) /
           (np.sum(ground_truth) + np.sum(prediction) + smooth))

    return acc



if __name__ == '__main__':
  pass

