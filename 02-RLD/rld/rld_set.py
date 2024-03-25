import os.path

import numpy as np


from tframe import console, pedia
from tframe import DataSet
from tframe import Predictor
from xomics import MedicalImage
from xomics.objects.general_mi import GeneralMI


class RLDSet(DataSet):

  def __init__(self, mi_data, buffer_size=None,
               name='dataset', data_dict=None):
    # super().__init__(**kwargs)
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
                          name=self.name + '(slice)', data_dict=self.data_dict)
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
                           name=self.name, data_dict=self.data_dict),
            self.__class__(mi_data=self.mi_data[sub_ids],
                           buffer_size=self.buffer_size,
                           name=name, data_dict=self.data_dict))

  @property
  def size(self): return len(self)

  def gen_random_window(self, batch_size):
    from rld_core import th
    from utils.data_processing import gen_windows
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    features, targets = gen_windows(self.features, self.targets, batch_size,
                                    th.window_size, th.slice_size)

    data_dict = {
      'features': np.array(features),
      'targets': np.array(targets)
    }
    if th.gan:
      data_dict[pedia.G_input] = data_dict.get('features')
      data_dict[pedia.D_input] = data_dict.get('targets')
    data_batch = DataSet(data_dict=data_dict)
    return data_batch

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    from rld_core import th
    if not is_training:
      if self.name == 'Test-Set':
        # assert self.size % batch_size == 0
        for i in range(self.size//batch_size+(self.size % batch_size)):
          features = self.features[i*batch_size:(i+1)*batch_size]
          targets = self.targets[i*batch_size:(i+1)*batch_size]
          data_dict = {
            'features': features,
            'targets': targets
          }
          if th.gan:
            data_dict[pedia.G_input] = data_dict.get('features')
            data_dict[pedia.D_input] = data_dict.get('targets')
          eval_set = DataSet(name=self.name, data_dict=data_dict)
          yield eval_set
        return
      else:
        index = list(np.random.choice(list(range(self.features.shape[0])),
                                      batch_size, replace=False))
        features, targets = self.features[index], self.targets[index]
        data_dict = {
          'features': features,
          'targets': targets
        }
        if th.gan:
          data_dict[pedia.G_input] = data_dict.get('features')
          data_dict[pedia.D_input] = data_dict.get('targets')
      eval_set = DataSet(name=self.name, data_dict=data_dict)
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

    dirpath = os.path.join(th.job_dir, 'checkpoints/', th.mark, 'saves/')
    pred = []
    if not update_saves and os.path.exists(dirpath):
      for path in os.listdir(dirpath):
        if path.endswith(r".nii.gz"):
          pred.append(RLDSet.load_nii(os.path.join(dirpath, path)))
    else:
      os.makedirs(dirpath, exist_ok=True)
      pred = []
      pred_raw = model.predict(self, batch_size=1)[:, ..., -1]
      # pred_raw = np.zeros((5, 263, 440, 440, 1))
      print(pred_raw.shape)
      # Remove negative values
      pred_raw[pred_raw < 0] = 0
      for num, sub, pred_i in zip(range(len(self)), self.pid, pred_raw):
        pred_path = os.path.join(dirpath, f'{num}-{sub}-pred.nii.gz')
        index = self.mi_data.index(sub)
        pred_i = self.mi_data.reverse_norm_suv(pred_i, index)
        GeneralMI.write_img(pred_i, pred_path, self.images.itk[index])
        pred.append(pred_i)

    self.pred = pred
    pred = np.stack(pred, axis=0)
    pred = np.expand_dims(pred, axis=-1)

    # statistics
    if th.statistics:
      stat_path = os.path.join(dirpath, 'stat/')
      if not os.path.exists(stat_path):
        os.makedirs(stat_path)
      self.evaluate_statistic(stat_path)

    if th.gen_test_nii:
      test_keys = ['30G', '240G', '40S', '60G', '240S', 'CT_seg']
      self.gen_test_data(dirpath, test_keys)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'{self.pid[i]}', images={
        'Input': self.images_raw[i],
        'Full': self.labels_raw[i],
        'Output': pred[i, ..., 0],
      }) for i in range(self.size)]

    if th.show_weight_map:
      wm_dir = os.path.join(dirpath, 'wm/')
      wm_path = os.path.join(wm_dir, 'wm.pkl')
      fetchers = [th.depot['weight_map']]
      if 'candidate1' in th.depot:
        fetchers.append(th.depot['candidate1'])
      if os.path.exists(wm_dir):
        wms = load(wm_path)
        shape = wms.shape[:-1] + (wms.shape[-1] - 1,)
        values = np.zeros(shape)
        c = values.shape[-1]
        c_tmp = 0
        p_tmp = 0
        for path in os.listdir(wm_dir):
          if path.endswith('.nii.gz'):
            values[p_tmp, ..., c_tmp] = self.load_nii(os.path.join(wm_dir, path))
            c_tmp = (c_tmp + 1) % c
            p_tmp = p_tmp+1 if c_tmp == 0 else p_tmp
      else:
        os.makedirs(wm_dir, exist_ok=True)
        wms, values = model.evaluate(fetchers, self, batch_size=1)

        max_wms = np.max(wms)
        min_wms = np.min(wms)
        wms = 255 * (wms - min_wms) / (max_wms - min_wms)
        wms = wms.astype(np.uint8)

        dump(wms, wm_path)
        rev_values = []
        for index, value in enumerate(values):
          tmp = []
          for v in range(1, value.shape[-1]):
            rev_v = self.mi_data.reverse_norm_suv(value[:, ..., v], index)
            tmp.append(rev_v)
            GeneralMI.write_img(rev_v, os.path.join(wm_dir, f'{index}{v}-ca{v}-{self.pid[index]}.nii.gz'),
                                self.images.itk[index])
          rev_values.append(np.stack(tmp, axis=-1))
        values = rev_values
        values = np.stack(values, axis=0)

      # wm.shape = [?, S, H, W, C]
      for wm, mi in zip(wms, medical_images):
        mi.put_into_pocket('weight_map', wm)

      for ca, mi in zip(values, medical_images):
        for c in range(ca.shape[-1]):
          mi.images[f'Candidate-{c}'] = ca[:, :, :, c]

    re = RLDExplorer(medical_images)
    self.mi_data.clean_mem()
    re.sv.set('vmin', auto_refresh=False)
    re.sv.set('vmax', auto_refresh=False)
    re.sv.set('full_key', 'Full')
    re.sv.set('cmap', 'gist_yarg')
    re.sv.set('vmax', 5.0)
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

    if len(th.extra_data) > 0:
      features = [self.images[subjects]]
      features += [self.mi_data.images[k][subjects] for k in th.extra_data]
      features = [np.expand_dims(np.stack(feature, axis=0), axis=-1)
                  for feature in features]
      features = np.concatenate(features, axis=-1)
    else:
      features = self.images[subjects]
      features = np.expand_dims(np.stack(features, axis=0), axis=-1)

    targets = self.labels[subjects]
    segs = self.seg[subjects]

    if th.use_seg is not None:
      onehot = np.zeros_like(features, dtype=bool)
      for i, seg in enumerate(segs):
        onehot[i] = np.expand_dims(self.mi_data.mask2onehot(seg, th.use_seg),
                                   axis=-1)
      features = np.concatenate([features, onehot], axis=-1)

    if not th.noCT:
      ct = self.mi_data.images['CT'][subjects]
      cts = np.expand_dims(np.stack(ct, axis=0), axis=-1)
      # print(cts.shape, features.shape)
      features = np.concatenate([features, cts], axis=-1)



    self.features = features
    self.targets = np.expand_dims(np.stack(targets, axis=0), axis=-1)

  def gen_test_data(self, dirpath, keys):
      self.mi_data.image_keys = keys
      for pid, seg in zip(self.pid, self.seg):
        data_path = os.path.join(dirpath, 'raw_data/')
        if not os.path.exists(data_path):
          os.makedirs(data_path)
        for name in self.mi_data.image_keys:
          img_path = os.path.join(data_path, f'{pid}-{name}.nii.gz')
          if os.path.exists(img_path): continue
          index = self.mi_data.index(pid)
          GeneralMI.write_img(self.mi_data.images_raw[name][index],
                              img_path, self.images.itk[index])

  def evaluate_statistic(self, path):
    from utils.statistics import load_suv_stat, draw_one_bar, set_ax, \
      load_metric_stat, hist_joint, violin_plot, violin_plot_roi, metric_text
    import matplotlib.pyplot as plt
    console.show_status(r'Calculating the Statistics...')

    input_label = '30s Gated'
    output_label = 'Predicted'
    true_label = '240s Gated'

    # metrics calc
    console.supplement(r'Calc the Metrics', level=2)
    metrics = ['SSIM', 'NRMSE', 'RELA', 'PSNR']
    input_metric = load_metric_stat(self.labels_raw, self.images_raw, metrics,
                                    path, self.pid, 'low')
    output_metric = load_metric_stat(self.labels_raw, self.pred, metrics,
                                     path, self.pid, 'predict')
    # SUV calc
    console.supplement(r'Calc the SUV', level=2)
    roi = [5, 10, 11, 12, 13, 14, 51]
    suv_max_input, suv_mean_input = load_suv_stat(self.images_raw, self.seg,
                                                  path, self.pid, 'low')
    suv_max_pred, suv_mean_pred = load_suv_stat(self.pred, self.seg,
                                                path, self.pid, 'predict')
    suv_max_full, suv_mean_full = load_suv_stat(self.labels_raw, self.seg,
                                                path, self.pid, 'full')
    # Pics Draw
    width = 0.3
    metric_x = np.arange(len(metrics))
    region_x = np.arange(len(roi))
    console.show_status(r'Start to draw the figure...')
    fig, axs = plt.subplots(2, 5, figsize=(32, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    # metric draw
    console.supplement(r'Draw the Metrics', level=2)
    axs[0, 0].bar(metric_x[:-1] - width/2, input_metric[0][:-1], width, label=input_label)
    axs[0, 0].errorbar(metric_x[:-1] - width / 2, input_metric[0][:-1],
                       yerr=input_metric[1][:-1], fmt='.', color='red',
                       ecolor='black', capsize=6)

    axs[0, 0].bar(metric_x[:-1] + width/2, output_metric[0][:-1], width, label=output_label)
    axs[0, 0].errorbar(metric_x[:-1] + width / 2, output_metric[0][:-1],
                       yerr=output_metric[1][:-1], fmt='.', color='red',
                       ecolor='black', capsize=6)

    ax_psnr = axs[0, 0].twinx()
    ax_psnr.bar(metric_x[-1] - width / 2, input_metric[0][-1],
                width, label=input_label)
    ax_psnr.errorbar(metric_x[-1] - width / 2, input_metric[0][-1],
                     yerr=input_metric[1][-1],
                     fmt='.', color='red', ecolor='black', capsize=6)

    ax_psnr.bar(metric_x[-1] + width / 2, output_metric[0][-1],
                width, label=output_label)
    ax_psnr.errorbar(metric_x[-1] + width / 2, output_metric[0][-1],
                     yerr=output_metric[1][-1],
                     fmt='.', color='red', ecolor='black', capsize=6)

    metric_text(axs[0, 0], ax_psnr, input_metric, metric_x, width)
    metric_text(axs[0, 0], ax_psnr, output_metric, metric_x, -width)

    axs[0, 0].set_xticks(metric_x, metrics)
    axs[0, 0].legend()
    axs[0, 0].set_title('Metrics')
    # hist joint draw
    console.supplement(r'Draw the Histogram', level=2)
    hist_joint(fig, axs[0, 1], self.images_raw, self.labels_raw,
               input_label, true_label, -3, 3)
    hist_joint(fig, axs[0, 2], self.pred, self.labels_raw,
               output_label, true_label, -3, 3)
    # suv draw
    console.supplement(r'Draw the SUV', level=2)
    draw_one_bar(axs[1, 0], region_x - width, suv_max_input, width, roi, input_label)
    draw_one_bar(axs[1, 0], region_x, suv_max_pred, width, roi, output_label)
    draw_one_bar(axs[1, 0], region_x + width, suv_max_full, width, roi, true_label)

    draw_one_bar(axs[1, 1], region_x - width, suv_mean_input, width, roi, input_label)
    draw_one_bar(axs[1, 1], region_x, suv_mean_pred, width, roi, output_label)
    draw_one_bar(axs[1, 1], region_x + width, suv_mean_full, width, roi, true_label)

    set_ax([axs[1, 0], axs[1, 1]], ['$SUV_{max}$', '$SUV_{mean}$'], region_x, roi)
    # violin draw
    console.supplement(r'Draw the Violin', level=2)
    violin_plot_roi(axs[1, 2], self.images_raw, self.seg, roi)
    violin_plot_roi(axs[0, 3], self.labels_raw, self.seg, roi)
    violin_plot_roi(axs[1, 3], self.pred, self.seg, roi)
    set_ax([axs[1, 2], axs[0, 3], axs[1, 3]],
           [input_label, true_label, output_label],
           np.arange(1, len(roi) + 1), roi, legend=False)

    violin_plot(axs[0, 4], [self.images_raw, self.pred, self.labels_raw], self.seg,
                5, [input_label, output_label, true_label])
    violin_plot(axs[1, 4], [self.images_raw, self.pred, self.labels_raw], self.seg,
                51, [input_label, output_label, true_label])

    fig.show()
    fig.savefig(os.path.join(path, 'figures.svg'), dpi=600, format='svg')

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
