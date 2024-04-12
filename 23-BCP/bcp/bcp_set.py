import random
import numpy as np
import os
import SimpleITK as sitk

from tframe import DataSet, Predictor
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

          feature = np.squeeze(mi_list[num].images['mr'])
          target = np.squeeze(mi_list[num].labels['caudate'])

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
          features.append(np.squeeze(mi_list[i].images['mr']))
          targets.append(np.squeeze(mi_list[i].labels['caudate']))

        features = np.expand_dims(features, axis=-1)
        targets = np.expand_dims(targets, axis=-1)
        name = 'batch_val'
        data_batch = DataSet(features=features, targets=targets, name=name)
        yield data_batch

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()


  def _check_data(self):
    pass


  def test_model(self, model):
    save_dir = r'E:\xai-omics\data\05-Brain-MR\yaojie_prediction'
    mi_list = self.data_dict['mi_list'].tolist()
    features, targets = [], []
    crop_infos = []
    pids = []
    for mi in mi_list:
      features.append(np.squeeze(mi.images['mr']))
      targets.append(np.squeeze(mi.labels['caudate']))
      crop_infos.append(mi._pocket['crop_info'])
      pids.append(mi.key)


    features = np.expand_dims(features, axis=-1)
    targets = np.expand_dims(targets, axis=-1)
    name = 'batch_eval'
    eval_set = DataSet(features=features, targets=targets, name=name)

    assert isinstance(model, Predictor)
    prediction = model.predict(eval_set, 16)
    prediction = np.squeeze(prediction)

    dice_score_list = []
    for pred, pid, crop_info, t in zip(prediction, pids, crop_infos, targets):
      dice_score = dice_accuarcy(pred, np.squeeze(t))
      dice_score_list.append(dice_score)
      print(f'{pid} : {dice_score}')
      save_path = os.path.join(save_dir, pid + '_pred.nii.gz')
      bottom, top = crop_info[0], crop_info[1]
      top = [256 - t for t in top]
      pred = np.pad(pred, ((bottom[0], top[0]), (bottom[1], top[1]), (bottom[2], top[2])))

      pred_image = sitk.GetImageFromArray(pred)
      sitk.WriteImage(pred_image, save_path)

    print(f'{self.name}: {np.mean(dice_score_list)}\n')


def dice_accuarcy(ground_truth, prediction):
  assert ground_truth.shape == prediction.shape
  smooth = 1.0

  intersection = np.sum(ground_truth * prediction)
  acc = ((2.0 * intersection + smooth) /
         (np.sum(ground_truth) + np.sum(prediction) + smooth))

  return acc



if __name__ == '__main__':
  pass

