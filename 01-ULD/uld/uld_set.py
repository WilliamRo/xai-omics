import os
import numpy as np


from tframe import console
from tframe import DataSet
from tframe import Predictor
from xomics.data_io.reader.uld_reader import UldReader


class ULDSet(DataSet):

  def __init__(self, data_dir=None, dose=None, buffer_size=None,
               subjects=None, name=None, data_dict=None):
    # super().__init__(**kwargs)
    from os import listdir
    self.data_dir = data_dir
    self.buffer_size = buffer_size
    self.subjects = []
    if subjects is None:
      for i in listdir(data_dir):
        if 'subject' in i:
          self.subjects.append(int(i[7:]))
    else:
      self.subjects = subjects
    self.data_fetcher = self.fetch_data
    self.reader: UldReader = UldReader(self.data_dir)
    self.dose = dose
    self.name = name

    self.data_dict = {} if data_dict is None else data_dict
    # Necessary fields to prevent errors
    self.is_rnn_input = False

  def gen_random_window(self, batch_size):
    from uld_core import th
    from utils.data_processing import gen_windows
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    if th.classify:
      index = np.random.randint(self.features.shape[0], size=batch_size)
      features, targets = self.features[index], self.targets[index]
    else:
      features, targets = gen_windows(self.features, self.targets, batch_size,
                                      th.window_size, th.slice_size, th.rand_batch)

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

  def classify_eval(self, model: Predictor):
    testpath = '../../../../data/01-ULD/testset/'
    doses = [
      'Full', '1-2', '1-4',
      '1-10', '1-20', '1-50', '1-100',
    ]
    reader = self.reader.load_as_npy_data(testpath, list(range(1, 51)),
                                          ('Anonymous_', '.nii.gz'),
                                          shape=(32, 160, 160))
    # with open(testpath+'tags.txt', 'rb') as f:
    #   tags = f.readlines()
    # tags = [i.split(b' ') for i in tags]

    raw_data = np.stack(reader.data)
    # suv_data = []
    #
    # for i in range(50):
    #   suv_data.append(raw_data[i] *
    #                   get_suv_factor(float(tags[i][1]), float(tags[i][0])))
    # suv_data = np.stack(suv_data)
    raw_data = raw_data.reshape(raw_data.shape + (1,))

    data = DataSet(raw_data, np.zeros((50, 7)))
    pred = model.predict(data)
    dose_pred = [doses[np.where(i == np.max(i))[0][0]] for i in pred]

    for i, j in enumerate(dose_pred):
      print(i+1, j)
    return

  def output_results(self, model: Predictor):
    from xomics.data_io.utils.raw_rw import wr_file
    from uld_core import th

    classification = {
      '1-4': [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
      '1-10': [2, 7, 12, 17, 22, 27, 32, 37, 42, 47],
      '1-20': [3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
      '1-50': [4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
      '1-100': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }

    testpath = '../../../../data/01-ULD/testset/'
    subs = classification[th.dose]
    reader = self.reader.load_as_npy_data(testpath, subs,
                                          ('Anonymous_', '.nii.gz'), raw=True,
                                          shape=[704, 440, 440])
    raw_data = reader.data
    sizes = reader.size_list
    param = reader.param_list
    for i, img in enumerate(raw_data):
      img = img.reshape((1, 704, 440, 440, 1))
      img_i = img / np.max(img)
      data_i = DataSet(img_i, img_i)
      pred = model.predict(data_i)
      pred_o = pred * np.max(img)
      pred_o = pred_o[0, :sizes[i][0], :sizes[i][1], :sizes[i][2], 0]
      pred_o.reshape((sizes[i][0], sizes[i][1], sizes[i][2]))
      filename = f'/outputs/Anonymous_{subs[i]}.nii.gz'
      wr_file(pred_o, testpath + filename, nii_param=param[i])
      print(f'({i+1}/{len(subs)}) saved the {filename}')

    return

  def evaluate_model(self, model: Predictor, report_metric=True):
    from uld_core import th
    from utils.metrics_calc import get_metrics
    import joblib

    if th.classify:
      return self.classify_eval(model)
    if th.output_result:
      return self.output_results(model)
    from dev.explorers.uld_explorer.uld_explorer_v31 import ULDExplorer
    from xomics import MedicalImage

    # if report_metric: model.evaluate_model(self, batch_size=1)
    # pred.shape = [N, s, s, s, 1]

    # from utils.split_combined import SplCom
    # splcom = SplCom(self.features, 11, [0, 24, 0, 0, 0])
    # for i, arr in enumerate(splcom):
    #   data = DataSet(arr, arr)
    #   console.supplement(f'model predicting({i+1}/{len(splcom)})', level=2)
    #   splcom.combine(model.predict(data))
    # pred = splcom.com_arr


    if not os.path.exists(th.mark+'.out'):
      data = DataSet(self.features, self.targets)
      pred = model.predict(data)
      joblib.dump(pred, th.mark + '.out')
    else:
      pred = joblib.load(th.mark + '.out')

    metrics = ['SSIM', 'NRMSE', 'PSNR']
    pmetric = get_metrics(self.targets[0, ..., 0],
                          pred[0, ..., 0],
                          metrics, data_range=1)

    lmetric = get_metrics(self.targets[0, ..., 0],
                          self.features[0, ..., 0],
                          metrics, data_range=1)
    console.show_status('metrics (low/predict)')
    for i in lmetric.keys():
      console.supplement(f'{i}:{lmetric[i]: .5f} /{pmetric[i]: .5f}', level=2)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'Sample-{i}', images={
        'Input': self.features[i],
        'Full': self.targets[i],
        'Model-Output': pred[i],
        # 'Delta': np.square(pred[i] - data.targets[i])
      }) for i in range(self.size)]

    if th.show_weight_map:
      fetchers = [th.depot['weight_map']]
      if 'candidate1' in th.depot: fetchers.append(th.depot['candidate1'])

      if not os.path.exists(th.mark+'_wm.out'):
        values = model.evaluate(fetchers, data)
        joblib.dump(values, th.mark + '_wm.out')
      else:
        values = joblib.load(th.mark+'_wm.out')

      wms = values[0]
      # wm.shape = [?, S, H, W, C]
      for wm, mi in zip(wms, medical_images):
        mi.put_into_pocket('weight_map', wm)

      if len(values) > 1:
        for ca, mi in zip(values[1], medical_images):
          for c in range(1, ca.shape[-1]):
            mi.images[f'Candidate-{c}'] = ca[:, :, :, c:c+1]



    ue = ULDExplorer(medical_images)
    ue.dv.set('vmin', auto_refresh=False)
    ue.dv.set('vmax', auto_refresh=False)

    ue.show()

  @staticmethod
  def fetch_data(self):
    return self._fetch_data()

  def _fetch_data(self):
    from uld_core import th
    if self.buffer_size is None or self.buffer_size >= len(self.subjects):
      subjects = self.subjects
    else:
      subjects = list(np.random.choice(self.subjects, self.buffer_size,
                                       replace=False))
    console.show_status(f'Fetching data subjects from {self.data_dir} ...')

    kwargs = {
      'use_suv': th.use_suv,
      'clip': (0, th.max_clip),
      'cmap': th.color_map,
      'shape': th.data_shape,
      'norm_margin': [0, 10, 0, 0, 0]
    }

    if th.classify:
      # kwargs['raw'] = True
      kwargs['show_log'] = False
      # kwargs['use_suv'] = True
      self._fetch_data_for_classify(subjects, **kwargs)
    elif th.norm_by_feature:
      doses = [[self.dose], ['Full']]
      data = self.reader.load_data(subjects, doses, methods='pair', **kwargs)
      self.features = np.concatenate(data['features'])
      self.targets = np.concatenate(data['targets'])
    elif th.train_self:
      data = self.reader.load_data(subjects, [[self.dose]], methods='sub', **kwargs)
      data = np.concatenate(data.values())
      self.targets = self.targets = data
    else:
      data = self.reader.load_data(subjects, [[self.dose], ['Full']],
                                   methods='type', **kwargs)
      self.features = np.concatenate(data[self.dose].values())
      self.targets = np.concatenate(data['Full'].values())

  def _fetch_data_for_classify(self, subjects, **kwargs):
    from tframe import pedia
    from tframe.utils import misc
    doses = [
      'Full', '1-2', '1-4',
      '1-10', '1-20', '1-50', '1-100',
    ]
    pedia.classe = doses
    self.NUM_CLASSES = 7
    doses = [[i] for i in doses]

    arr = []
    label = []

    for subject in subjects:
      data = self.reader.load_data([subject], doses, methods='type', **kwargs)
      data = list(data.values())
      data = [i[0] for i in data]
      # vmax = np.max(data[6])
      data = np.concatenate(data)
      arr.append(data)
      # arr.append(data / vmax)
      label += [[0], [1], [2], [3], [4], [5], [6]]
    self.features = np.concatenate(arr)
    self.targets = np.array(label)

    self.targets = misc.convert_to_one_hot(self.targets, self.NUM_CLASSES)


  @classmethod
  def load_as_uldset(cls, data_dir, dose=None):
    from tframe import hub as th
    return ULDSet(data_dir=data_dir, dose=dose,
                  buffer_size=th.buffer_size)


  def get_subsets(self, *sizes, names):
    # TODO: improve the function like split
    sizes = list(sizes)
    autos = sizes.count(-1)
    assert len(sizes) == 3
    assert autos <= 1

    num = np.sum(sizes) + autos
    if num > len(self.subjects):
      num = len(self.subjects) - 2
      sizes = [-1, 1, 1]

    index = sizes.index(-1)
    # results = random.sample(self.subjects, num)
    results = list(range(1, num+1))
    sub_list = []
    for k in range(len(sizes)):
      if k != index:
        item = self.__class__(self.data_dir, self.dose, self.buffer_size,
                              results[:sizes[k]], names[k])
        results = results[sizes[k]:]
      else:
        for i in results:
          self.subjects.remove(i)
        self.name = names[index]
        item = self
      sub_list.append(item)

    return sub_list


  def snapshot(self, model):
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt
    from utils.metrics_calc import get_metrics
    assert isinstance(model, Predictor)


    slice_num = 64
    if model.counter == 50:
      metrics = ['SSIM', 'NRMSE', 'PSNR', 'RMSE']
      fmetric = get_metrics(self.targets[0, ..., 0],
                            self.features[0, ..., 0],
                            metrics, data_range=1)
      fm_str = '-'.join([f'{k}{v:.5f}' for k, v in fmetric.items()])
      ffn = f'Feature-Slice{slice_num}-{fm_str}.png'
      tfn = f'Target-Slice{slice_num}.png'
      feature = self.features[0, slice_num, ..., 0]
      target = self.targets[0, slice_num, ..., 0]
      vmax = np.max(target)
      self.vmax = vmax
      plt.imsave(os.path.join(model.agent.ckpt_dir, ffn),
                 feature, cmap='gray', vmin=0., vmax=vmax)
      plt.imsave(os.path.join(model.agent.ckpt_dir, tfn),
                 target, cmap='gray', vmin=0., vmax=vmax)
    # (1) Get image (shape=[1, S, H, W, 1])
    data = DataSet(self.features[:1], self.targets[:1])
    images = model.predict(data)

    # (2) Get metrics
    val_dict = model.validate_model(data)

  # (3) Save image
    metric_str = '-'.join([f'{k}{v:.5f}' for k, v in val_dict.items()])
    fn = f'Iter{model.counter}-{metric_str}.png'
    img = images[0, slice_num, ..., 0]
    plt.imsave(os.path.join(model.agent.ckpt_dir, fn),
               img, cmap='gray', vmin=0., vmax=self.vmax)
