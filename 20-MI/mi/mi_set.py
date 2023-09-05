import random
import numpy as np
import os

from tframe import DataSet
from roma import console
from tools import data_processing
from tqdm import tqdm
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from xomics.objects import MedicalImage



class MISet(DataSet):

  def report(self):
    pass


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    from mi_core import th
    data_num = 10
    # if is_training and callable(self.data_fetcher):
    #   self.data_fetcher(self)
    mi_list = self.fetch_data(data_num=data_num, is_training=is_training)

    round_len = self.get_round_length(batch_size, training=is_training)

    if is_training:
      for i in range(round_len):
        features, targets = [], []
        for j in range(batch_size):
          num = random.randint(0, data_num - 1)
          mi = mi_list[num]

          # Preprocessing
          mi.window('ct', th.window[0], th.window[1])
          mi.normalization(['ct', 'pet'])
          mi.crop(th.crop_size)

          # Data Augmentation
          if th.random_flip:
            for axe in range(3):
              if random.choice([True, False]):
                data_processing.mi_flip(mi, axes=axe)

          if th.random_rotation:
            data_processing.mi_rotation(
              mi, angle=random.choice([0, 90, 180, 270]))

          if th.random_noise:
            data_processing.mi_add_gaussian_noise(mi)

          # whether to use pet data as the second channel
          if th.use_pet:
            features.append(
              np.stack([mi.images['ct'], mi.images['pet']], axis=-1))
          else:
            features.append(np.expand_dims(mi.images['ct'], axis=-1))
          targets.append(np.expand_dims(mi.labels['label-0'], axis=-1))

        name = 'batch_train_' + str(i)
        data_batch = MISet(
          features=np.array(features), targets=np.array(targets), name=name)
        yield data_batch
    else:
      # number = list(range(self.size))
      number = list(range(len(mi_list)))
      number_list = [number[i:i+th.val_batch_size]
                     for i in range(0, len(number), th.val_batch_size)]

      for num in number_list:
        features, targets = [], []
        for i in num:
          # mi: MedicalImage = MedicalImage.load(mi_file_list[i])
          mi = mi_list[i]

          # Preprocessing
          mi.window('ct', th.window[0], th.window[1])
          mi.normalization(['ct', 'pet'])
          mi.crop(th.crop_size)

          if th.use_pet:
            features.append(
              np.stack([mi.images['ct'], mi.images['pet']], axis=-1))
          else:
            features.append(np.expand_dims(mi.images['ct'], axis=-1))
          targets.append(np.expand_dims(mi.labels['label-0'], axis=-1))

        name = 'batch_val'
        data_batch = MISet(
          features=np.array(features), targets=np.array(targets), name=name)
        yield data_batch

    if is_training: self._clear_dynamic_round_len()


  def fetch_data(self, data_num, is_training: bool):
    mi_list = []
    mi_file_list = self.data_dict['mi_file_list'].tolist()
    if is_training or 'Train' in self.name or 'train' in self.name:
      for file in random.sample(mi_file_list, data_num):
        console.show_status(
          'Loading `{}` ... from {}'.format(file, self.name))
        mi_list.append(MedicalImage.load(file))
    else:
      for file in mi_file_list:
        console.show_status(
          'Loading `{}` ... from {}'.format(file, self.name))
        mi_list.append(MedicalImage.load(file))

    return mi_list


  def visualize_self(self, example_num):
    '''
    visualize self
    example_num means the number of the examples
    '''
    assert example_num > 0

    from mi_core import th

    example_num = self.size if example_num > self.size else example_num
    mi_file_list = self.data_dict['mi_file_list']
    mi_list = []

    for i in range(example_num):
      mi: MedicalImage = MedicalImage.load(mi_file_list[i])

      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct', 'pet'])

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
    from mi_core import th
    results = model.predict(self)
    results[results <= 0.5] = 0
    results[results > 0.5] = 1

    mi_file_list = self.data_dict['mi_file_list']

    assert len(mi_file_list) == results.shape[0]

    mi_list = []
    for i, file in enumerate(mi_file_list):
      mi: MedicalImage = MedicalImage.load(file)

      # Preprocessing
      mi.window('ct', th.window[0], th.window[1])
      mi.normalization(['ct', 'pet'])
      mi.crop(th.crop_size)

      mi.labels['prediction'] = np.squeeze(results[i])
      mi_list.append(mi)

    self.visulization(mi_list)

    print()



if __name__ == '__main__':
  from mi_core import th
  from mi.mi_agent import MIAgent

  data = MIAgent.load_as_tframe_data(th.data_dir)

  data.visualize_self(5)

  print()


