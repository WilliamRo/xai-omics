import numpy as np
import os


from tframe import console, pedia
from tframe import DataSet
from tframe import Predictor
from xomics import MedicalImage
from xomics.objects.jutils.objects import GeneralMI


class RLDSet(DataSet):

  # region: Basic methods

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
    self.properties = {}

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

  # endregion: Basic methods

  # region: Data batch relevant methods

  def gen_random_window(self, batch_size):
    from rld_core import th
    from utils.data_processing import gen_windows
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    if th.windows_size is not None:
      features, targets = gen_windows(self.features, self.targets, batch_size,
                                      th.windows_size)
    else:
      index = np.random.choice(list(range(self.features.shape[0])), batch_size,
                               replace=False)
      features, targets = self.features[index], self.targets[index]

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
          if self.batch_preprocessor is not None:
            eval_set = self.batch_preprocessor(eval_set, is_training)
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
      if self.batch_preprocessor is not None:
        eval_set = self.batch_preprocessor(eval_set, is_training)
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
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch, is_training)
      # Yield data batch
      yield data_batch
    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()

  # endregion: Data batch relevant methods

  # region: Evaluation methods

  def generate_demo(self, model, **kwargs):
    from pictor import Pictor
    from tframe.utils import imtool

    def plot(fig, x): imtool.gan_grid_plot(x, fig=fig)

    p = Pictor('DDPM Demo')
    p.add_plotter(plot)
    p.objects = model.generate(delta_t=0, x_T=self.targets[200:201],
                               return_all_images=True, **kwargs)
    p.show()

  def evaluate_model(self, model: Predictor, report_metric=False, update_saves=False):
    from dev.explorers.rld_explore.rld_explorer import RLDExplorer
    from rld_core import th
    from joblib import load, dump

    dirpath = os.path.join(th.job_dir, 'checkpoints/', th.mark, 'saves/')
    pred = []
    if not update_saves and os.path.exists(dirpath):
      for path in sorted(os.listdir(dirpath)):
        if path.endswith(r".nii.gz"):
          pred.append(RLDSet.load_nii(os.path.join(dirpath, path)))
    else:
      os.makedirs(dirpath, exist_ok=True)
      pred = []
      pred_raw = []
      for i in range(len(self)):
        datadict = {
          'features': self.features[i:i+1],
          'targets': self.targets[i:i+1]
        }
        if th.gan:
          datadict[pedia.G_input] = datadict.get('targets')
        data = DataSet(data_dict=datadict)
        pred_raw.append(model.predict(data, batch_size=1)[:, ..., -1])
      pred_raw = np.concatenate(pred_raw, axis=0)
      if th.dimension == 2:
        pred_raw = np.reshape(pred_raw, [-1]+th.data_shape)
      # pred_raw = np.zeros((5, 263, 440, 440))
      print(pred_raw.shape)
      # Remove negative values
      pred_raw[pred_raw < 0] = 0
      for num, sub, pred_i in zip(range(len(self)), self.pid, pred_raw):
        pred_path = os.path.join(dirpath, f'{num}-{sub}-pred.nii.gz')
        index = self.mi_data.index(sub)
        print(np.max(pred_i), np.min(pred_i))
        pred_i = self.mi_data.reverse_norm_suv(pred_i, index)
        GeneralMI.write_img(pred_i, pred_path, self.itk_imgs[th.data_set[0]][index])
        pred.append(pred_i)

    # self.pred = pred
    # pred = np.stack(pred, axis=0)
    # pred = np.expand_dims(pred, axis=-1)

    if th.gen_gaussian != 0:
      data_path = os.path.join(dirpath, 'gaussian/')
      if not os.path.exists(data_path):
        os.makedirs(data_path)
      for i in range(len(self)):
        index = self.mi_data.index(self.pid[i])
        raw = self.features[i, ..., 0]
        img = self.gen_gaussian(raw, th.gen_gaussian)
        img = self.mi_data.reverse_norm_suv(img, i)
        GeneralMI.write_img(img, os.path.join(data_path,
                                              f'{i}-{self.pid[i]}-pred.nii.gz'),
                            self.itk_imgs[th.data_set[0]][index])

    if th.gen_test_nii:
      # test_keys = th.data_set[:1]
      test_keys = ['CT']
      self.gen_test_data(dirpath, test_keys)

    if th.gen_mask:
      self.gen_mask(dirpath)

    if th.gen_dcm:
      dcm_data = pred[:, ..., 0]
      dcm_path = os.path.join(dirpath, 'dcm/')
      if not os.path.exists(dcm_path):
        os.makedirs(dcm_path)
      for i in range(len(self)):
        index = self.mi_data.index(self.pid[i])
        tags = self.mi_data.get_tags(i)
        dcm = self.mi_data.suv_reverse(dcm_data[i], tags)
        GeneralMI.write_img_dcm(dcm, os.path.join(dcm_path, f'{i}-{self.pid[i]}.dcm'),
                                tags, self.itk_imgs[th.data_set[0]][index])
      pass

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'{self.pid[i]}', images={
        'Input': self.raw_images[th.data_set[0]][i],
        'Full': self.raw_images[th.data_set[1]][i],
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
                                self.images[th.data_set[0]].itk[index])
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

  def gen_mask(self, dirpath):
    from rld_core import th
    data_path = os.path.join(dirpath, 'mask/')
    if not os.path.exists(data_path):
      os.makedirs(data_path)
    for i in range(len(self)):
      pid = self.pid[i]
      img_path = os.path.join(data_path, f'{i}-{pid}-mask.nii.gz')
      if os.path.exists(img_path): continue
      index = self.mi_data.index(pid)
      seg = self.seg[i]
      if pid == 'YHP00012417':
        seg = seg[::-1]
      GeneralMI.write_img(seg, img_path,
                          self.itk_imgs[th.data_set[0]][index])

  def gen_test_data(self, dirpath, keys):
    from rld_core import th
    self.mi_data.image_keys = keys
    data_path = os.path.join(dirpath, 'raw_data/')
    if not os.path.exists(data_path):
      os.makedirs(data_path)
    for i in range(len(self)):
      pid = self.pid[i]
      for name in self.mi_data.image_keys:
        img_path = os.path.join(data_path, f'{i}-{pid}-{name}.nii.gz')
        if os.path.exists(img_path): continue
        index = self.mi_data.index(pid)
        if name != 'CT':
          GeneralMI.write_img(self.raw_images[name][index][0],
                              img_path, self.itk_imgs[th.data_set[0]][index])
        else:
          ct_img = self.raw_images[name][index][0]
          if pid == 'YHP00012417':
            ct_img = ct_img[::-1]
          GeneralMI.write_img(ct_img, img_path,
                              self.itk_imgs[th.data_set[0]][index])

  def gen_gaussian(self, img, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma)

  # endregion: Evaluation methods

  # region: Data Fetch methods

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
      features = [self.images[th.data_set[0]][subjects]]
      features += [self.images[k][subjects] for k in th.extra_data]
      features = [np.expand_dims(np.stack(feature, axis=0), axis=-1)
                  for feature in features]
      features = np.concatenate(features, axis=-1)
    else:
      features = self.images[th.data_set[0]][subjects]
      features = np.expand_dims(np.stack(features, axis=0), axis=-1)

    targets = self.images[th.data_set[1]][subjects]

    if th.use_seg is not None:
      segs = self.seg[subjects]
      onehot = np.zeros_like(features, dtype=bool)
      for i, seg in enumerate(segs):
        onehot[i] = np.expand_dims(self.mi_data.mask2onehot(seg, th.use_seg),
                                   axis=-1)
      features = np.concatenate([features, onehot], axis=-1)

    if not th.noCT:
      ct = self.images['CT'][subjects]
      cts = np.expand_dims(np.stack(ct, axis=0), axis=-1)
      # print(cts.shape, features.shape)
      features = np.concatenate([features, cts], axis=-1)



    self.features = features
    self.targets = np.expand_dims(np.stack(targets, axis=0), axis=-1)

    if th.dimension == 2:
      self.features = np.reshape(self.features, [-1]+th.data_shape[1:])
      self.targets = np.reshape(self.targets, [-1]+th.data_shape[1:])
      self.features = np.expand_dims(self.features, axis=-1)
      self.targets = np.expand_dims(self.targets, axis=-1)

    if th.ddpm:
      self.features, self.targets = self.targets, self.features

  # endregion: Data Fetch methods

  # region: Properties and static methods

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
    return self.mi_data.images

  @property
  def raw_images(self):
    return self.mi_data.raw_images

  @property
  def itk_imgs(self):
    return self.mi_data.itk_imgs

  @property
  def itk_raws(self):
    return self.mi_data.itk_raws

  @staticmethod
  def load_nii(filepath):
    from xomics.data_io.utils.raw_rw import rd_file
    return rd_file(filepath)

  @staticmethod
  def export_nii(data, filepath, **kwargs):
    from xomics.data_io.utils.raw_rw import wr_file
    return wr_file(data, filepath, **kwargs)

  # endregion: Properties and static methods