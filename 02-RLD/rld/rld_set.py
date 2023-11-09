import numpy as np


from tframe import console
from tframe import DataSet
from tframe import Predictor
from xomics import MedicalImage
from xomics.data_io.reader.rld_reader import RLDReader


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
      for i in sorted(listdir(data_dir)):
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
    if isinstance(item, str):
      if item in self.data_dict.keys():
        return self.data_dict[item]
      elif item in self.properties.keys():
        return self.properties[item]
      else:
        raise KeyError('!! Can not resolve "{}"'.format(item))

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
                                    th.window_size, th.slice_size)

    data_batch = DataSet(features, targets)

    return data_batch

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    if not is_training:
      index = list(np.random.choice(list(range(self.features.shape[0])),
                                    batch_size, replace=False))
      features, targets = self.features[index], self.targets[index]
      eval_set = DataSet(features, targets, name=self.name)
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
    from dev.explorers.rld_explore.rld_explorer import RLDExplorer
    from xomics.data_io.utils.raw_rw import wr_file

    if report_metric:
      model.evaluate_model(self, batch_size=2)
    pred = model.predict(self, batch_size=2)

    for i in range(self.size):
      wr_file(pred[i][:, ..., 0], f'sub-{self.subjects[i]}-pred.nii.gz',
              self.reader.img_param)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'sub-{self.subjects[i]}', images={
        'Input': self.features[i][:, ..., 1:],
        'Full': self.targets[i],
        'Output': pred[i],
      }) for i in range(self.size)]

    re = RLDExplorer(medical_images)
    re.sv.set('vmin', auto_refresh=False)
    re.sv.set('vmax', auto_refresh=False)
    re.show()

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
      'norm_margin': [10, 0, 0]
    }

    types = [
      ['PET', 'WB', '30S', 'GATED'],
      ['PET', 'WB', '240S', 'GATED'],
      ['CT', 'WB'],
    ]

    if th.noCT:
      types = types[:-1]
      f_index = [0]
    else:
      f_index = [0, len(types)-1]
    t_index = [1]

    data = self.reader.load_data(subjects, types, methods='train', **kwargs)
    features, targets = [], []
    for i in range(len(data)):
      imgs = np.stack(data[i].images.values())
      if i in f_index:
        features.append(imgs)
      elif i in t_index:
        targets.append(imgs)
    self.features = np.concatenate(features, axis=-1)
    self.targets = np.concatenate(targets, axis=-1)


