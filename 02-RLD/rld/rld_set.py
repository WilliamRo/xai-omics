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

    self._pred = None

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
        pred_i = self.mi_data.reverse_norm_suv(pred_i, index)
        GeneralMI.write_img(pred_i, pred_path, self.images.itk[index])
        pred.append(pred_i)
      if report_metric:
        metric = model.evaluate_model(self, batch_size=1)
        dump([pred, metric], cache_path)

    self.pred = pred
    pred = np.stack(pred, axis=0)
    pred = np.expand_dims(pred, axis=-1)

    features = self.images_raw
    targets = self.labels_raw
    if th.gen_test_nii:
      for pid, feature, target, seg in zip(self.pid, features, targets, self.seg):
        data_path = os.path.join(dirpath, 'raw_data/')
        if not os.path.exists(data_path):
          os.makedirs(data_path)
        low_path = os.path.join(data_path, f'{pid}-low.nii.gz')
        full_path = os.path.join(data_path, f'{pid}-full.nii.gz')
        seg_path = os.path.join(data_path, f'{pid}-seg.nii.gz')
        index = self.mi_data.index(pid)
        GeneralMI.write_img(feature, low_path, self.images.itk[index])
        GeneralMI.write_img(target, full_path, self.images.itk[index])
        GeneralMI.write_img(self.seg, seg_path, self.images.itk[index])

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'{self.pid[i]}', images={
        'Input': self.images_raw[i],
        'Full': self.labels_raw[i],
        'Output': pred[i, ..., 0],
      }) for i in range(self.size)]

    self.evaluate_statistic()

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

  def evaluate_statistic(self):
    from utils.statistics import calc_suv_statistic, draw_one_bar, set_ax, \
      get_mean_std_metric, hist_joint, violin_plot, violin_plot_roi
    import matplotlib.pyplot as plt
    console.show_status(r'Calculating the Statistica...')
    fig, axs = plt.subplots(2, 5, figsize=(12, 22))
    # metrics calc
    metrics = ['SSIM', 'NRMSE', 'RELA', 'PSNR']
    input_metric = get_mean_std_metric(self.labels_raw, self.images_raw, metrics)
    output_metric = get_mean_std_metric(self.labels_raw, self.pred, metrics)
    # SUV calc
    roi = [5, 10, 11, 12, 13, 14, 51]
    suv_max_input, suv_mean_input = calc_suv_statistic(self.images_raw,
                                                       self.seg, roi)
    suv_max_pred, suv_mean_pred = calc_suv_statistic(self.pred, self.seg, roi)
    suv_max_full, suv_mean_full = calc_suv_statistic(self.labels_raw,
                                                     self.seg, roi)
    # Pics Draw
    width = 0.3
    metric_x = np.arange(len(metrics))
    region_x = np.arange(len(roi))

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # metric draw
    axs[0, 0].bar(metric_x[:-1] - width/2, input_metric[0][:-1], width, label='30s Gated')
    axs[0, 0].errorbar(metric_x[:-1] - width / 2, input_metric[0][:-1],
                       yerr=input_metric[1][:-1], fmt='.', color='red',
                       ecolor='black', capsize=6)

    axs[0, 0].bar(metric_x[:-1] + width/2, output_metric[0][:-1], width, label='Predicted')
    axs[0, 0].errorbar(metric_x[:-1] + width / 2, output_metric[0][:-1],
                       yerr=output_metric[1][:-1], fmt='.', color='red',
                       ecolor='black', capsize=6)

    ax_psnr = axs[0, 0].twinx()
    ax_psnr.bar(metric_x[-1] - width / 2, input_metric[0][-1],
                width, label='30s Gated')
    ax_psnr.errorbar(metric_x[-1] - width / 2, input_metric[0][-1],
                     yerr=input_metric[1][-1],
                     fmt='.', color='red', ecolor='black', capsize=6)

    ax_psnr.bar(metric_x[-1] + width / 2, output_metric[0][-1],
                width, label='30s Gated')
    ax_psnr.errorbar(metric_x[-1] + width / 2, output_metric[0][-1],
                     yerr=output_metric[1][-1],
                     fmt='.', color='red', ecolor='black', capsize=6)
    axs[0, 0].set_xticks(metric_x, metrics)
    axs[0, 0].set_title('Metrics')
    # hist joint draw
    hist_joint(fig, axs[0, 1], self.images_raw, self.labels_raw,
               '30s Gated', '240s Gated', -3, 3)
    hist_joint(fig, axs[0, 2], self.pred, self.labels_raw,
               'Predicted', '240s Gated', -3, 3)
    # suv draw
    draw_one_bar(axs[1, 0], region_x - width, suv_max_input, width, roi, '30s Gated')
    draw_one_bar(axs[1, 0], region_x, suv_max_pred, width, roi, 'Predicted')
    draw_one_bar(axs[1, 0], region_x + width, suv_max_full, width, roi, '240s Gated')

    draw_one_bar(axs[1, 1], region_x - width, suv_mean_input, width, roi, '30s Gated')
    draw_one_bar(axs[1, 1], region_x, suv_mean_pred, width, roi, 'Predicted')
    draw_one_bar(axs[1, 1], region_x + width, suv_mean_full, width, roi, '240s Gated')

    set_ax([axs[1, 0], axs[1, 1]], ['$SUV_{max}$', '$SUV_{mean}$'], region_x, roi)
    # violin draw
    violin_plot_roi(axs[1, 2], self.images_raw, self.seg, roi)
    violin_plot_roi(axs[0, 3], self.labels_raw, self.seg, roi)
    violin_plot_roi(axs[1, 3], self.pred, self.seg, roi)
    set_ax([axs[1, 2], axs[0, 3], axs[1, 3]],
           ['30s Gated', '240s Gated', 'Predicted'],
           np.arange(1, len(roi) + 1), roi, legend=False)

    violin_plot(axs[0, 4], [self.images_raw, self.pred, self.labels_raw], self.seg,
                5, ['30s Gated', 'Predicted', '240s Gated'])
    violin_plot(axs[1, 4], [self.images_raw, self.pred, self.labels_raw], self.seg,
                51, ['30s Gated', 'Predicted', '240s Gated'])

    fig.show()

  @property
  def pid(self):
    return self.mi_data.pid

  @property
  def seg(self):
    return self.mi_data.images['CT_seg']

  @property
  def pred(self):
    return self._pred

  @pred.setter
  def pred(self, value):
    self._pred = value

  @pred.getter
  def pred(self):
    assert self._pred is not None
    return self._pred

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
