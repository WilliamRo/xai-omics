import random
import numpy as np
import os

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

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, self.size - 1)

          feature = np.squeeze(self.features[num])
          target = np.squeeze(self.targets[num])

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
          features.append(np.squeeze(self.features[i]))
          targets.append(np.squeeze(self.targets[i]))

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
    from bcp_core import th
    mi_dir = os.path.join(
      os.path.dirname(os.path.dirname(th.data_dir)),
      'data/05-Brain-MR/mi/NFBS')
    file_name = os.listdir(mi_dir)
    file_name = [f for f in file_name if '.mi' in f]
    ss = model.validate_model(self, th.eval_batch_size)
    sss = model.predict(self, th.eval_batch_size)
    acc = dice_accuarcy(self.targets, sss)

    mi_list = []
    for f in tqdm(file_name, desc='Loading MI files'):
      mi = MedicalImage.load(os.path.join(mi_dir, f))
      mi.normalization(['mr'], 'min_max')
      mi.images['mr'] = mi.images['mr'] * 2 - 1

      mi.crop([mi.shape[0], th.xy_size, th.xy_size], False, [])

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

      feature = np.expand_dims(feature, axis=-1)
      target = np.expand_dims(target, axis=-1)

      eval_set = BCPSet(features=feature, targets=target, name='EvalSet')

      assert isinstance(model, Predictor)
      prediction = model.predict(eval_set, 16)

      prediction = np.squeeze(prediction)


      slice_num = prediction.shape[1]
      result_z_shape = prediction.shape[0] + slice_num - 1
      result_xy_shape = prediction.shape[-1]
      result = np.zeros([result_z_shape, result_xy_shape, result_xy_shape])
      add_count = np.zeros(result_z_shape)

      for i, p in enumerate(prediction):
        result[i: i + slice_num] = result[i: i + slice_num] + p
        add_count[i: i + slice_num] = add_count[i: i + slice_num] + 1

      for i, a in enumerate(add_count):
        result[i] = result[i] / a

      result[result >= 0.5] = 1
      result[result < 0.5] = 0

      pred = np.zeros_like(mi.images['mr'])
      pred[np.min(indices): np.max(indices) + 1] = result

      mi.labels['pred'] = pred
      acc = dice_accuarcy(mi.labels["label-0"], mi.labels["pred"])
      mi.key = mi.key + f' --- Acc: {round(acc, 2)}'
      print(mi.key)

      mi_list.append(mi)

    dg = DrGordon(mi_list)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.show()


def dice_accuarcy(ground_truth, prediction):
  assert ground_truth.shape == prediction.shape
  smooth = 1.0

  intersection = np.sum(ground_truth * prediction)
  acc = ((2.0 * intersection + smooth) /
         (np.sum(ground_truth) + np.sum(prediction) + smooth))

  return acc



if __name__ == '__main__':
  pass

