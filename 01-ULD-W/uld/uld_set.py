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

    self.properties = {}

  def gen_random_window(self, batch_size):
    from uld_core import th
    from utils.data_processing import gen_windows
    # Randomly sample [S, S, S] pair from features and targets

    if th.classify:
      index = np.random.randint(self.features.shape[0], size=batch_size)
      features, targets = self.features[index], self.targets[index]
    else:
      features, targets = gen_windows(self.features, self.targets, batch_size,
                                      th.window_size, th.slice_size, th.rand_batch)

    data_batch = DataSet(features, targets)

    return data_batch



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
      'norm_margin': [0, 10, 0, 0, 0],
    }

    if th.classify:
      kwargs['raw'] = True
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
      vmax = np.max(data[6])
      arr.append(data / vmax)
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
    results = self.subjects[:num]
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



  def evaluate_model(self, model: Predictor, report_metric=True):
    from dev.explorers.uld_explorer.uld_explorer_v31 import ULDExplorer
    from uld_core import th
    from xomics import MedicalImage

    if report_metric: model.evaluate_model(self, batch_size=1)

    # pred.shape = [N, s, s, s, 1]
    data = DataSet(self.features, self.targets)
    # data = self.gen_random_window(128)
    pred = model.predict(data)
    print(self.size, pred.shape, self.targets.shape)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'Sample-{i}', images={
        'Input': data.features[i],
        'Full': data.targets[i],
        'Model-Output': pred[i],
        # 'Delta': np.square(pred[i] - data.targets[i])
      }) for i in range(self.size)]

    if th.show_weight_map:
      fetchers = [th.depot['weight_map']]
      if 'candidate1' in th.depot: fetchers.append(th.depot['candidate1'])

      values = model.evaluate(fetchers, data)
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
    ue.dv.set('axial_margin', 0, auto_refresh=False)

    ue.show()
