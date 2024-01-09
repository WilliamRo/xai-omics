import os.path

import numpy as np


from tframe import console
from tframe import DataSet
from tframe import Predictor
from xomics import MedicalImage
from xomics.data_io.reader.general_mi import GeneralMI


class RLDSet(DataSet):

  def __init__(self, mi_data, buffer_size=None,
               name='dataset', data_dict=None):
    # super().__init__(**kwargs)
    from os import listdir
    self.mi_data: GeneralMI = mi_data
    self.buffer_size = buffer_size
    self.data_fetcher = self.fetch_data
    self.name = name

    self.data_dict = {} if data_dict is None else data_dict
    # Necessary fields to prevent errors
    self.is_rnn_input = False

  def __len__(self): return len(self.mi_data)

  def __getitem__(self, item):
    # If item is index array
    if isinstance(item, str):
      if item in self.data_dict.keys():
        return self.data_dict[item]
      elif item in self.properties.keys():
        return self.properties[item]
      else:
        raise KeyError('!! Can not resolve "{}"'.format(item))

    data_set = type(self)(mi_data=self.mi_data[item],
                          buffer_size=self.buffer_size,
                          name=self.name + '(slice)')
    return data_set

  def subset(self, pids, name=None):
    if name is None:
      name = self.name + '(slice)'
    sub_ids = []
    ids = []
    for i in pids:
      sub_ids.append(self.mi_data.index(i))
    for i in range(len(self)):
      if i not in sub_ids:
        ids.append(i)
    return (self.__class__(mi_data=self.mi_data[ids],
                           buffer_size=self.buffer_size,
                           name=self.name),
            self.__class__(mi_data=self.mi_data[sub_ids],
                           buffer_size=self.buffer_size,
                           name=name))

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
        # assert self.size % batch_size == 0
        for i in range(self.size//batch_size+(self.size % batch_size)):
          features = self.features[i*batch_size:(i+1)*batch_size]
          targets = self.targets[i*batch_size:(i+1)*batch_size]
          eval_set = DataSet(features, targets, name=self.name)
          yield eval_set
        return
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

  def evaluate_model(self, model: Predictor, report_metric=False, update_saves=False):
    from dev.explorers.rld_explore.rld_explorer import RLDExplorer
    from rld_core import th
    from joblib import load, dump
    from xomics.data_io.utils.metrics_calc import calc_metric

    dirpath = os.path.join(th.job_dir, 'checkpoints/', th.mark, 'saves/')
    cache_path = os.path.join(dirpath, 'caches.pkl')
    pred = []
    if not update_saves and os.path.exists(dirpath):
      if report_metric:
        metric = load(cache_path)
        console.supplement(f'Metric:{metric}', level=2)
      for path in os.listdir(dirpath):
        if path.endswith('.nii.gz'):
          pred.append(RLDSet.load_nii(os.path.join(dirpath, path)))
    else:
      os.makedirs(dirpath, exist_ok=True)
      pred = []
      pred_tmp = model.predict(self, batch_size=1)[:, ..., -1]
      # pred_tmp = np.zeros((5, 263, 440, 440, 1))
      print(pred_tmp.shape)
      for num, sub, pred_i in zip(range(len(self)), self.pid, pred_tmp):
        pred_path = os.path.join(dirpath, f'{num}-{sub}-pred.nii.gz')
        index = self.mi_data.index(sub)
        pred_i = pred_i * np.max([self.mi_data.images_raw[0][index],
                                  self.mi_data.labels_raw[0][index]])
        GeneralMI.write_img(pred_i, pred_path, self.images.itk[index])
        pred.append(pred_i)
      if report_metric:
        metric = model.evaluate_model(self, batch_size=1)
        dump([pred, metric], cache_path)

    pred = np.stack(pred, axis=0)
    pred = np.expand_dims(pred, axis=-1)

    features = self.images_raw
    targets = self.labels_raw
    if th.gen_test_nii:
      for pid, feature, target in zip(self.pid, features, targets):
        data_path = os.path.join(dirpath, 'raw_data/')
        if not os.path.exists(data_path):
          os.makedirs(data_path)
        low_path = os.path.join(data_path, f'{pid}-low.nii.gz')
        full_path = os.path.join(data_path, f'{pid}-full.nii.gz')
        index = self.mi_data.index(pid)
        GeneralMI.write_img(feature, low_path, self.images.itk[index])
        GeneralMI.write_img(target, full_path, self.images.itk[index])

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'{self.pid[i]}', images={
        'Input': self.images_raw[i],
        'Full': self.labels_raw[i],
        'Output': pred[i, ..., 0],
      }) for i in range(self.size)]

    if th.show_weight_map:
      value_path = os.path.join(dirpath, 'wm.pkl')
      fetchers = [th.depot['weight_map']]
      if 'candidate1' in th.depot:
        fetchers.append(th.depot['candidate1'])
      if os.path.exists(value_path):
        values = load(value_path)
      else:
        values = model.evaluate(fetchers, self, batch_size=1)
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
    if self.buffer_size is None or self.buffer_size >= len(self):
      subjects = list(range(self.size))
    else:
      subjects = list(np.random.choice(range(len(self)), self.buffer_size,
                                       replace=False))
    console.show_status(f'Fetching data from {th.data_kwargs["dataset"]} ...')

    features = self.images[subjects]
    targets = self.labels[subjects]

    features = np.expand_dims(np.stack(features, axis=0), axis=-1)
    if not th.noCT:
      ct = self.mi_data.images['CT'][subjects]
      cts = np.expand_dims(np.stack(ct, axis=0), axis=-1)
      # print(cts.shape, features.shape)
      features = np.concatenate([features, cts], axis=-1)

    self.features = features
    self.targets = np.expand_dims(np.stack(targets, axis=0), axis=-1)

  @property
  def pid(self):
    return self.mi_data.pid

  @property
  def images(self):
    return self.mi_data.images[0]

  @property
  def labels(self):
    return self.mi_data.labels[0]

  @property
  def images_raw(self):
    return self.mi_data.images_raw[0]

  @property
  def labels_raw(self):
    return self.mi_data.labels_raw[0]

  @staticmethod
  def load_nii(filepath):
    from xomics.data_io.utils.raw_rw import rd_file
    return rd_file(filepath)

  @staticmethod
  def export_nii(data, filepath, **kwargs):
    from xomics.data_io.utils.raw_rw import wr_file
    return wr_file(data, filepath, **kwargs)
