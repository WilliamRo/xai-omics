import random
from os import listdir

import numpy as np
from tframe import console
from tframe import DataSet
from tframe import Predictor
from utils.data_processing import gen_windows
from xomics.data_io.mi_reader import load_data


class ULDSet(DataSet):

  def __init__(self, data_dir=None, dose=None, buffer_size=None,
               subjects=None, name=None, data_dict=None):
    # super().__init__(**kwargs)
    self.data_dir = data_dir
    self.buffer_size = buffer_size
    self.subjects = []
    if subjects is None:
      for i in listdir(data_dir):
        if 'subject' in i:
          self.subjects.append(i)
    else:
      self.subjects = subjects
    self.data_fetcher = self.fetch_data
    self.dose = dose
    self.name = name

    self.data_dict = {} if data_dict is None else data_dict
    # Necessary fields to prevent errors
    self.is_rnn_input = False

  def gen_random_window(self, batch_size):
    from uld_core import th
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    s = th.window_size
    features, targets = gen_windows(self.features, self.targets, batch_size,
                                    s, th.slice_size)

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
    from xomics.gui.dr_gordon import DrGordon
    from xomics import MedicalImage

    if report_metric: model.evaluate_model(self, batch_size=1)

    # pred.shape = [N, s, s, s, 1]
    data = DataSet(self.features, self.targets)
    # data = self.gen_random_window(128)
    pred = model.predict(data, batch_size=1)
    print(self.size, pred.shape, self.targets.shape)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'Sample-{i}', images={
        'Input': data.features[i],
        'Targets': data.targets[i],
        'Model-Output': pred[i]})
      for i in range(self.size)]

    dg = DrGordon(medical_images)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.show()

  @staticmethod
  def fetch_data(self):
    if self.buffer_size is None or self.buffer_size >= len(self.subjects):
      subjects = self.subjects
    else:
      subjects = np.random.choice(self.subjects, self.buffer_size, replace=False)
    console.show_status(f'Fetching signal groups to {self.data_dir} ...')
    self.features = load_data(self.data_dir, subjects, self.dose)
    self.targets = load_data(self.data_dir, subjects, "Full")


  @classmethod
  def load_as_uldset(cls, data_dir, dose):
    from tframe import hub as th
    return ULDSet(data_dir=data_dir, dose=dose,
                  buffer_size=th.buffer_size)


  def get_subsets(self, *sizes, names):
    # TODO: improve the function like split
    if len(sizes) != 3:
      raise SystemError("The function is need to upgrade!")
    results = random.sample(self.subjects, sizes[1]+sizes[2])
    for i in results:
      self.subjects.remove(i)
    self.name = names[0]

    return self, self.__class__(self.data_dir, self.dose, self.buffer_size,
                                results[:sizes[1]], names[1]), \
        self.__class__(self.data_dir, self.dose, self.buffer_size,
                       results[-sizes[2]:], names[2])


  def snapshot(self, model):
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)

    # (1) Get image (shape=[1, S, H, W, 1])
    data = DataSet(self.features[:1, 280:340], self.targets[:1, 280:340])
    images = model.predict(data)

    # (2) Get metrics
    val_dict = model.validate_model(self)

    # (3) Save image
    metric_str = '-'.join([f'{k}{v:.2f}' for k, v in val_dict.items()])
    fn = f'Iter{model.counter}-{metric_str}.png'
    plt.imsave(os.path.join(model.agent.ckpt_dir, fn),
               images[0, 320, ..., 0], cmap='gray')
