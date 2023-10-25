import os
import numpy as np

from tframe import console
from tframe import DataSet
from tframe import Predictor
from xomics import MedicalImage
from xomics.data_io.rld_reader import RLDReader


class RLDSet(DataSet):

  def __init__(self, data_dir=None, buffer_size=None, subjects=None,
               name='dataset', data_dict=None):
    # super().__init__(**kwargs)
    from os import listdir
    self.data_dir = data_dir
    self.buffer_size = buffer_size
    self.data_fetcher = self.fetch_data
    self.reader: RLDReader = RLDReader(self.data_dir)
    self.name = name

    self.subjects = []
    if subjects is None:
      for i in listdir(data_dir):
        if i.startswith('sub'):
          self.subjects.append(int(i[3:]))
    else:
      self.subjects = subjects

    self.data_dict = {} if data_dict is None else data_dict
    # Necessary fields to prevent errors
    self.is_rnn_input = False

  def __len__(self): return len(self.subjects)

  def __getitem__(self, item):
    # If item is index array
    assert type(item) == slice
    data_set = type(self)(subjects=self.subjects[item],
                          buffer_size=self.buffer_size,
                          data_dir=self.data_dir,
                          name=self.name + '(slice)')
    return data_set

  @property
  def size(self): return len(self)

  def gen_random_window(self, batch_size):
    from rld_core import th
    from utils.data_processing import gen_windows
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    features, targets = gen_windows(self.features, self.targets, batch_size,
                                    th.window_size, th.slice_size, False)

    data_batch = DataSet(features, targets)

    return data_batch

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    if not is_training:
      index = np.random.randint(self.features.shape[0], size=batch_size)
      features, targets = self.features[index], self.targets[index]
      eval_set = DataSet(features, targets, name=self.name + '-Eval')
      yield eval_set
      return
    elif callable(self.data_fetcher):
      self.data_fetcher(self)
    round_len = self.get_round_length(batch_size, training=is_training)
    # Generate batches
    # !! Without `if is_training:`, error will occur if th.validate_train_set
    #    is on
    for i in range(round_len):
      data_batch = self.gen_random_window(batch_size)
      # Yield data batch
      yield data_batch
    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()

  def evaluate_model(self, model: Predictor, report_metric=True):

    pred = model.predict(self)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'Sample-{i}', images={
        'Input': self.features[i],
        'Full': self.targets[i],
        'Model-Output': pred[i],
        # 'Delta': np.square(pred[i] - data.targets[i])
      }) for i in range(self.size)]

    # re = RLDExplorer(medical_images)
    # re.dv.set('vmin', auto_refresh=False)
    # re.dv.set('vmax', auto_refresh=False)
    # re.show()

  @staticmethod
  def fetch_data(self):
    return self._fetch_data()

  def _fetch_data(self):
    from rld_core import th
    if self.buffer_size is None or self.buffer_size >= len(self.subjects):
      subjects = self.subjects
    else:
      subjects = list(np.random.choice(self.subjects, self.buffer_size,
                                       replace=False))
    console.show_status(f'Fetching data from {self.data_dir} ...')

    kwargs = {
      'norm_types': ['PET'],
      'use_suv': th.use_suv,
      'clip': th.data_clip,
      'shape': th.data_shape,
      'norm_margin': [0, 10, 0, 0, 0]
    }

    types = [
      ['CT', 'WB'],
      ['PET', 'WB', '240S', 'GATED'],
      ['PET', 'WB', '240S', 'STATIC'],
    ]

    f = lambda x: np.concatenate(data['_'.join(types[x])])

    data = self.reader.load_data(subjects, types, methods='train', **kwargs)
    self.features = np.concatenate([f(0), f(1)], axis=4)
    self.targets = f(2)
    pass

  @classmethod
  def load_as_rldset(cls, data_dir):
    from tframe import hub as th
    return RLDSet(data_dir=data_dir, buffer_size=th.buffer_size)
