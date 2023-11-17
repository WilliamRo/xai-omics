import os.path

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
      if self.name == 'Test-Set':
        features, targets = self.features, self.targets
      else:
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

  def evaluate_model(self, model: Predictor, report_metric=True, update_saves=False):
    from dev.explorers.rld_explore.rld_explorer import RLDExplorer
    from rld_core import th
    from joblib import load, dump
    from xomics.data_io.utils.raw_rw import wr_file
    from xomics.data_io.utils.metrics_calc import calc_metric

    dirpath = os.path.join(th.job_dir, 'checkpoints/', th.mark, 'saves/')
    cache_path = os.path.join(dirpath, 'caches.pkl')
    if not update_saves and os.path.exists(dirpath):
      if report_metric:
        pred, metric = load(cache_path)
        console.supplement(f'Metric:{metric}', level=2)
      else:
        pred = load(cache_path)
    else:
      os.makedirs(dirpath, exist_ok=True)
      pred = model.predict(self, batch_size=2)
      if report_metric:
        metric = model.evaluate_model(self, batch_size=2)
        dump([pred, metric], cache_path)
      else:
        dump(pred, cache_path)

    features, targets = self.features[:, ..., :1], self.targets
    for i in range(self.size):
      features[i] = self.reader.get_raw_data(self.features[i, ..., :1], i)
      targets[i] = self.reader.get_raw_data(self.targets[i, ..., :1], i)
      pred[i] = self.reader.get_raw_data(pred[i, ..., :1], i)

    # for i in range(self.size):
    #   ssim_ori = calc_metric(self.features[i, ..., 0], self.targets[i, ..., 0], 'ssim')
    #   ssim = calc_metric(self.features[i, ..., 0], pred[i, ..., 0], 'ssim')
    #   raw_data = self.reader.get_raw_data(pred[i, ..., 0], i)
    #   raw_data = np.clip(raw_data, a_min=0, a_max=None)
    #   wr_file(raw_data, f'sub{self.subjects[i]}-'
    #                     f'ssim({ssim_ori:.5f}-{ssim:.5f})'
    #                     f'-pred.nii.gz',
    #           self.reader.img_param[i])
    #   wr_file(self.features[i][::-1], f'sub{self.subjects[i]}-'
    #                                   f'feature.nii.gz',
    #           self.reader.img_param[i])
    #   wr_file(self.targets[i][::-1], f'sub{self.subjects[i]}-'
    #                                  f'target.nii.gz',
    #           self.reader.img_param[i])

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'sub-{self.subjects[i]}', images={
        'Input': features[i],
        'Full': targets[i],
        'Output': pred[i],
      }) for i in range(self.size)]

    if th.show_weight_map:
      value_path = os.path.join(dirpath, 'wm.pkl')
      fetchers = [th.depot['weight_map']]
      if 'candidate1' in th.depot:
        fetchers.append(th.depot['candidate1'])
      if os.path.exists(value_path):
        values = load(value_path)
      else:
        values = model.evaluate(fetchers, self)
        dump(values, value_path)

      wms = values[0]
      # wm.shape = [?, S, H, W, C]
      for wm, mi in zip(wms, medical_images):
        mi.put_into_pocket('weight_map', wm)

      if len(values) > 1:
        for ca, mi in zip(values[1], medical_images):
          for c in range(1, ca.shape[-1]):
            mi.images[f'Candidate-{c}'] = ca[:, :, :, c:c+1]

    re = RLDExplorer(medical_images)
    re.sv.set('vmin', auto_refresh=False)
    re.sv.set('vmax', auto_refresh=False)
    re.sv.set('full_key', 'Full')
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
      'suv': th.use_suv,
      'clip': th.data_clip,
      'shape': th.data_shape,
      'norm_margin': th.data_margin if th.train else None
    }

    types = [
      ['PET', 'WB', '20S', 'STATIC'],
      ['PET', 'WB', '30S', 'GATED'],
      ['PET', 'WB', '120S', 'STATIC'],
      ['PET', 'WB', '240S', 'GATED'],
      ['PET', 'WB', '240S', 'STATIC'],
      ['CT', 'WB'],
    ]

    types = [types[i] for i in th.data_set] + [types[-1]]

    if th.noCT:
      types = types[:-1]
      f_index = [0]
      kwargs['noCT'] = True
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


