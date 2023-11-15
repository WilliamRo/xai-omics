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



class ESSet(DataSet):

  def report(self):
    pass


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    from es_core import th

    round_len = self.get_round_length(batch_size, training=is_training)

    if is_training:
      classes = self.data_dict['class']
      region_index = list(np.where((classes == [0, 1]).all(axis=1))[0])
      lesion_index = list(np.where((classes == [1, 0]).all(axis=1))[0])
      if region_index == [] and lesion_index == []:
        assert TypeError
      elif region_index == []:
        index_list = [lesion_index, lesion_index]
      elif lesion_index == []:
        index_list = [region_index, region_index]
      else:
        index_list = [region_index, lesion_index]

      for i in range(round_len):
        features, targets, region_masks, classes = [], [], [], []
        index_all = [random.sample(index_list[b % 2], 1)[0]
                     for b in range(batch_size)]
        crop_size_list = [self.data_dict['crop_size'][ind] for ind in index_all]
        crop_size = [max(column) for column in zip(*crop_size_list)]

        for j in range(batch_size):
          index = index_all[j]

          feature = self.features[index]
          target = self.targets[index]
          region_mask = self.data_dict['region_mask'][index]
          cls = self.data_dict['class'][index]

          # transform images and labels to [S, H, W]
          ct = feature[:, :, :, 0]
          pet = feature[:, :, :, 1]
          target = target[:, :, :, 0]
          region_mask = region_mask[:, :, :, 0]

          # Data Augmentation
          basis = [target, region_mask] if np.array_equal(cls, [0, 1]) else [target]
          cropped_data = image_processing.crop_3d(
            [ct, pet, target, region_mask], crop_size,
            th.random_crop, basis)

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
            np.stack([ct, pet], axis=-1))
          targets.append(np.expand_dims(target, axis=-1))
          region_masks.append(np.expand_dims(region_mask, axis=-1))
          classes.append(cls)

        name = 'batch_train_' + str(i)
        data_dict = {
          'features': np.array(features),
          'targets': np.array(targets),
          'region_mask': np.array(region_masks),
          'lesion_mask': np.array(targets)
        }
        data_batch = DataSet(data_dict=data_dict, name=name)
        yield data_batch
    else:
      number = list(range(self.size))
      number_list = [number[i:i+th.val_batch_size]
                     for i in range(0, len(number), th.val_batch_size)]

      for num in number_list:
        features, targets, region_masks, classes = [], [], [], []
        # z_size = random.choice([128, 64, 32])
        # xy_size = random.choice([512, 256, 128, 64])
        # crop_size = [z_size, xy_size, xy_size]
        crop_size = [128, 256, 256]

        for i in num:
          feature = self.features[i]
          target = self.targets[i]
          region_mask = self.data_dict['region_mask'][i]
          cls = self.data_dict['class'][i]

          # transform images and labels to [S, H, W]
          ct = feature[:, :, :, 0]
          pet = feature[:, :, :, 1]
          target = target[:, :, :, 0]
          region_mask = region_mask[:, :, :, 0]

          # Data Augmentation
          basis = [target, region_mask] if np.array_equal(cls, [0, 1]) else [target]
          cropped_data = image_processing.crop_3d(
            [ct, pet, target, region_mask], crop_size, False, basis)

          ct, pet, target, region_mask = (cropped_data[0], cropped_data[1],
                                          cropped_data[2], cropped_data[3])

          # Features and Targets
          features.append(
            np.stack([ct, pet], axis=-1))
          targets.append(np.expand_dims(target, axis=-1))
          region_masks.append(np.expand_dims(region_mask, axis=-1))
          classes.append(cls)

        name = 'batch_val'
        data_dict = {
          'features': np.array(features),
          'targets': np.array(targets),
          'region_mask': np.array(region_masks),
          'lesion_mask': np.array(targets)
        }
        data_batch = DataSet(data_dict=data_dict, name=name)
        yield data_batch

    if is_training: self._clear_dynamic_round_len()


  def visualize_self(self, example_num):
    '''
    visualize self
    example_num means the number of the examples
    '''
    assert example_num > 0

    from es_core import th

    example_num = self.size if example_num > self.size else example_num
    mi_file_list = self.data_dict['mi_file_list']
    mi_list = []

    for i in range(example_num):
      mi: MedicalImage = MedicalImage.load(mi_file_list[i])

      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct'])

      mi_list.append(mi)

    dg = DrGordon(mi_list, title=self.name)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.show()


  def visulization(self, mi_list):
    dg = DrGordon(mi_list, title=self.name)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.show()


  def _check_data(self):
    pass


  def test_model(self, model):
    from es_core import th
    print(f'Test Model in {self.name}')

    pids = self.data_dict['pids'].tolist()

    features = np.squeeze(self.features)
    lesions = np.squeeze(self.targets)
    regions = np.squeeze(self.data_dict['region_mask'])
    classes = np.squeeze(self.data_dict['class'])

    assert len(features) == len(pids)

    predictions = model.predict(self)
    predictions = np.squeeze(predictions)
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    assert len(lesions) == len(predictions)

    mi_list = []
    save_dir = r'E:\xai-omics\data\02-PET-CT-Y1\results\04-prt\mi'
    model_file = r'1111_prt(8-5-2-1-relu-mp)_Sc_23'
    save_dir = os.path.join(save_dir, model_file)
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    for f, l, r, p, pid, cls in zip(
        features, lesions, regions, predictions, pids, classes):
      ct, pet = f[..., 0], f[..., 1]
      basis = [l, r] if np.array_equal(cls, [0, 1]) else [l]
      ct, pet, l = image_processing.crop_3d(
        [ct, pet, l], [128, 256, 256], False, basis)

      images = {'ct': ct, 'pet': pet}
      labels = {'Lesion/Ground Truth': l, 'Prediction': p}

      acc = self.dice_accuarcy(ground_truth=l, prediction=p)
      key = pid + f'--- Dice Acc: {round(acc, 2)}'
      print(key)

      mi: MedicalImage = MedicalImage(
        images=images, labels=labels, key=key)
      mi.save(os.path.join(save_dir, pid + '.mi'))
      mi_list.append(mi)

    # self.visulization(mi_list)


  def dice_accuarcy(self, ground_truth, prediction):
    assert ground_truth.shape == prediction.shape
    smooth = 1.0

    intersection = np.sum(ground_truth * prediction)
    acc = ((2.0 * intersection + smooth) /
           (np.sum(ground_truth) + np.sum(prediction) + smooth))

    return acc


def get_min_crop_size(input_datas: list):
  assert all(isinstance(arr, np.ndarray) for arr in input_datas)
  assert all(arr.shape == input_datas[0].shape for arr in input_datas)

  union = np.logical_or.reduce(input_datas)
  indices = np.argwhere(union == 1)
  max_indices = np.max(indices, axis=0)
  min_indices = np.min(indices, axis=0)

  delta_indices = [
    maxi - mini for maxi, mini in zip(max_indices, min_indices)]

  return [next_power_of_2(d) for d in delta_indices]


def next_power_of_2(number):
  if number == 0:
    return 0

  if number and not (number & (number - 1)):
    return number

  return 2 ** math.ceil(math.log2(number))



if __name__ == '__main__':
  pass

