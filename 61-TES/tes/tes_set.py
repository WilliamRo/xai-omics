import random
import numpy as np
import os

from tframe import DataSet, Predictor
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from tools import image_processing
from copy import copy


class TESSet(DataSet):


  def report(self):
    pass


  def _check_data(self):
    pass


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    from tes_core import th

    round_len = self.get_round_length(batch_size, training=is_training)

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, self.size - 1)
          feature = np.squeeze(self.features[num])
          target = np.squeeze(self.targets[num])

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
      for i in range(0, self.size, th.val_batch_size):
        features = self.features[i: i + th.val_batch_size]
        targets = self.targets[i: i + th.val_batch_size]
        name = 'batch_val'
        data_batch = DataSet(features=features, targets=targets, name=name)
        yield data_batch

    if is_training: self._clear_dynamic_round_len()


  def test_model(self, model):
    from tes_core import th
    # mi_dir = os.path.join(
    #   os.path.dirname(os.path.dirname(th.data_dir)),
    #   'data/02-PET-CT-Y1/mi/es')
    # file_name = ['0022_huyunda.mi', '0023_zhangbairong.mi']
    mi_dir = os.path.join(
      os.path.dirname(os.path.dirname(th.data_dir)),
      'data/02-PET-CT-Y1/Public Datasets/SegTHOR/mi')
    file_name = os.listdir(mi_dir)[:2]

    mi_list = []
    features = []
    targets = []
    pred_slice_num = []
    for f in tqdm(file_name, desc='Loading MI files'):
      mi = MedicalImage.load(os.path.join(mi_dir, f))
      mi.window('ct', th.window[0], th.window[1])
      mi.crop([mi.shape[0], th.xy_size, th.xy_size], random_crop=False,
              basis=list(mi.labels.keys()))
      mi.normalization(['ct'], 'min_max')
      mi.images['ct'] = mi.images['ct'] * 2 - 1

      indices = np.where(mi.labels['label-0'] == 1)[0]
      mi.images['ct'] = mi.images['ct'][
                        np.min(indices): np.max(indices) + 1]
      mi.labels['label-0'] = mi.labels['label-0'][
                             np.min(indices): np.max(indices) + 1]

      mi_list.append(mi)

      image = mi.images['ct']
      feature = [image[i:i + th.slice_num, :, :]
                 for i in range(image.shape[0] - th.slice_num + 1)]
      label = mi.labels['label-0']
      target = [label[i:i + th.slice_num, :, :]
                for i in range(label.shape[0] - th.slice_num + 1)]
      features.append(np.squeeze(feature))
      targets.append(np.squeeze(target))
      pred_slice_num.append(len(feature))

    features = np.expand_dims(np.concatenate(features, axis=0), axis=-1)
    targets = np.expand_dims(np.concatenate(targets, axis=0), axis=-1)
    eval_set = TESSet(features=features, name='EvalSet')

    assert isinstance(model, Predictor)
    prediction = model.predict(eval_set, 16)
    print(f'Acc: {dice_accuarcy(targets, prediction)}')
    prediction = np.squeeze(prediction)

    sub_pred = []
    start_index = 0
    for length in tqdm(pred_slice_num, desc='Pred_slice_num - length'):
      sublist = prediction[start_index:start_index + length]
      sub_pred.append(sublist)
      start_index += length

    pred_all = []
    for pred in tqdm(sub_pred, desc='sub_pred - pred'):
      slice_num = pred.shape[1]
      result_z_shape = pred.shape[0] + slice_num - 1
      result_xy_shape = pred.shape[-1]
      result = np.zeros([result_z_shape, result_xy_shape, result_xy_shape])
      add_count = np.zeros(result_z_shape)

      for i, p in enumerate(pred):
        result[i: i + slice_num] = result[i: i + slice_num] + p
        add_count[i: i + slice_num] = add_count[i: i + slice_num] + 1

      for i, a in enumerate(add_count):
        result[i] = result[i] / a

      result[result >= 0.5] = 1
      result[result < 0.5] = 0
      pred_all.append(result)

    for mi, p in zip(mi_list, pred_all):
      mi.labels['pred'] = p
      acc = dice_accuarcy(mi.labels['label-0'], mi.labels['pred'])
      print(f'{mi.key} acc: {acc}')

    dg = DrGordon(mi_list)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.show()


  def predict_label(self, model):
    from tes_core import th

    import pickle
    pickle_file = os.path.join(
      os.path.dirname(os.path.dirname(th.data_dir)),
      'data/02-PET-CT-Y1/mi/crop_dict.pkl')
    with open(pickle_file, 'rb') as file:
      crop_dict = pickle.load(file)

    mi_dir = os.path.join(
      os.path.dirname(os.path.dirname(th.data_dir)),
      'data/02-PET-CT-Y1/mi/es_test')
    file_name = os.listdir(mi_dir)[:9]

    mi_list = []
    features = []
    pred_slice_num = []
    for f in tqdm(file_name, desc='Loading MI files'):
      mi = MedicalImage.load(os.path.join(mi_dir, f))
      mi.crop([mi.shape[0], th.xy_size, th.xy_size], random_crop=False,
              basis=list(mi.labels.keys()))
      mi.normalization(['ct'], 'z_score')

      indices = np.where(crop_dict[mi.key] == 1)[0]
      mi.images['ct'] = mi.images['ct'][np.min(indices): np.max(indices) + 1]
      mi_list.append(mi)

      image = mi.images['ct']
      feature = [image[i:i + th.slice_num, :, :]
                 for i in range(image.shape[0] - th.slice_num + 1)]
      features.append(np.squeeze(feature))
      pred_slice_num.append(len(feature))

    features = np.expand_dims(np.concatenate(features, axis=0), axis=-1)
    testset = TESSet(features=features, name='TESTSET')

    assert isinstance(model, Predictor)
    prediction = model.predict(testset, 16)
    prediction = np.squeeze(prediction)

    sub_pred = []
    start_index = 0
    for length in pred_slice_num:
      sublist = prediction[start_index:start_index + length]
      sub_pred.append(sublist)
      start_index += length


    pred_all = []
    for pred in sub_pred:
      slice_num = pred.shape[1]
      result_z_shape = pred.shape[0] + slice_num - 1
      result_xy_shape = pred.shape[-1]
      result = np.zeros([result_z_shape, result_xy_shape, result_xy_shape])
      add_count = np.zeros(result_z_shape)

      for i, p in enumerate(pred):
        result[i: i + slice_num] = result[i: i + slice_num] + p
        add_count[i: i + slice_num] = add_count[i: i + slice_num] + 1

      for i, a in enumerate(add_count):
        result[i] = result[i] / a

      result[result >= 0.5] = 1
      result[result < 0.5] = 0
      pred_all.append(result)

    for mi, p in zip(mi_list, pred_all):
      mi.labels['pred'] = p

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
  from tes.tes_agent import TESAgent
  train_set, _, _ = TESAgent.load()


  print()


