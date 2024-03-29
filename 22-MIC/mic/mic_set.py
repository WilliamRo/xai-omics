import random
import numpy as np
import os
import SimpleITK as sitk

from tframe import DataSet
from roma import console
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from tools import data_processing
from copy import deepcopy, copy



class MICSet(DataSet):

  @property
  def dataset_for_eval(self):
    from mic_core import th

    features, labels, patient_ids = [], [], []

    for mi in self.features.tolist():
      # Data Preprocessing
      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct'], 'z_score')
      mi.crop(th.crop_size, False, ['label-0'])

      labels.append(mi.labels['label-0'])
      patient_ids.append(mi.key)

      if th.use_mask:
        features.append(
          np.stack([mi.images['ct'], mi.labels['label-0']], axis=-1))
      else:
        features.append(np.expand_dims(mi.images['ct'], axis=-1))

    data_dict = {'features': np.array(features),
                 'targets': self.targets,
                 'labels': np.array(labels),
                 'patient_ids': patient_ids}

    return MICSet(data_dict=data_dict, name=self.name)


  def report(self):
    pass


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    '''

    '''
    from mic_core import th

    round_len = self.get_round_length(batch_size, training=is_training)

    if is_training:
      mi_list, labels = self.features.tolist(), self.targets

      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, len(mi_list) - 1)

          # Data Preprocessing
          mi: MedicalImage = copy(mi_list[num])
          mi.window('ct', th.window[0], th.window[1])
          mi.crop(th.crop_size, th.random_translation, ['label-0'])
          mi.normalization(['ct'], 'z_score')

          # Data Augmentation
          if th.random_flip:
            if random.choice([True, False]):
              data_processing.mi_flip(mi, axes=random.choice([0, 1, 2]))

          if th.random_rotation:
            data_processing.mi_rotation(
              mi, angle=random.choice([0, 90, 180, 270]))

          if th.random_noise:
            data_processing.mi_add_gaussian_noise(mi)

          if th.use_mask:
            features.append(
              np.stack([mi.images['ct'], mi.labels['label-0']], axis=-1))
          else:
            features.append(np.expand_dims(mi.images['ct'], axis=-1))
          targets.append(copy(labels[num]))

        name = 'batch_train_' + str(i)
        data_batch = DataSet(features=np.array(features),
                             targets=np.array(targets), name=name)
        yield data_batch
    elif 'Train' in self.name or 'train' in self.name:
      mi_list, labels = self.features.tolist(), self.targets

      number = list(range(len(mi_list)))
      number_list = [number[i:i+th.val_batch_size]
                     for i in range(0, len(number), th.val_batch_size)]

      for num in number_list:
        features, targets = [], []
        for i in num:
          mi: MedicalImage = copy(mi_list[i])

          # Data Preprocessing
          mi.window('ct', th.window[0], th.window[1])
          mi.crop(th.crop_size, False, ['label-0'])
          mi.normalization(['ct'], 'z_score')

          if th.use_mask:
            features.append(
              np.stack([mi.images['ct'], mi.labels['label-0']], axis=-1))
          else:
            features.append(np.expand_dims(mi.images['ct'], axis=-1))
          targets.append(copy(labels[i]))

        name = 'batch_val'
        data_batch = DataSet(features=np.array(features),
                             targets=np.array(targets), name=name)
        yield data_batch
    else:
      yield DataSet(
        features=self.features, targets=self.targets, name=self.name)

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()


  def fetch_data(self, data_num):
    mi_list, labels = [], []
    mi_file_list = self.data_dict['features'].tolist()

    data_num = min(data_num, self.size)

    BA_num = min(len(self.groups[0]), data_num // 2)
    MIA_num = data_num - BA_num

    BA_indice = random.sample(self.groups[0], BA_num)
    MIA_indice = random.sample(self.groups[1], MIA_num)

    for indice in BA_indice + MIA_indice:
      console.show_status(
        'Loading `{}`  from {}'.format(mi_file_list[indice], self.name))
      mi_list.append(MedicalImage.load(mi_file_list[indice]))
      labels.append(self.targets[indice])

    return mi_list, np.array(labels)


  def visualize_self(self, example_num):
    '''
    visualize self
    example_num means the number of the examples
    '''
    assert example_num > 0

    from mi_core import th

    example_num = self.size if example_num > self.size else example_num
    mi_file_list = self.data_dict['features']
    mi_list = []

    for i in range(example_num):
      mi: MedicalImage = MedicalImage.load(mi_file_list[i])

      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct'])
      mi.crop([32, 64, 64], False)
      mi.key = mi.key + ' --- {}'.format(
        self.properties['classes'][self.dense_labels[i]])

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
    results = model.predict(self)
    results[results < 0.5] = 0
    results[results >= 0.5] = 1

    indices_target = np.argmax(self.targets, axis=1)
    indices_prediction = np.argmax(results, axis=1)

    accuracy = (np.sum(indices_target == indices_prediction) /
                indices_target.shape[0])
    print('Accuarcy:', round(accuracy * 100, 2), '%')

    cancer_type = ['BA', 'MIA']
    patient_ids = self.data_dict['patient_ids']

    mi_list = []
    for i in range(self.size):
      mi: MedicalImage = MedicalImage(
        images={'ct': np.squeeze(self.features[i])},
        labels={'label-0': np.squeeze(self.data_dict['labels'][i])},
        key=f'PID: {patient_ids[i]} -- '
            f'Ground Truth: {cancer_type[indices_target[i]]} -- '
            f'Prediction: {cancer_type[indices_prediction[i]]}')
      mi_list.append(mi)

    self.visulization(mi_list)


    print()



if __name__ == '__main__':
  pass
