import numpy as np

from tframe import DataSet
from tframe import Predictor
from utils.data_processing import gen_windows


class ULDSet(DataSet):


  def gen_random_window(self, batch_size):
    from uld_core import th
    # Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    s = th.window_size
    features, targets = gen_windows(self.features, self.targets, batch_size, s)

    data_batch = DataSet(features, targets)

    return data_batch


  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    if not is_training:
      index = np.random.randint(self.features.shape[0], size=batch_size)
      features, targets = self.features[index], self.targets[index]
      eval_set = DataSet(features, targets, name=self.name + '-Eval')
      yield eval_set
      return
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
    pred = model.predict(data, batch_size=1)
    print(self.size, pred.shape, self.targets.shape)

    # Compare results using DrGordon
    medical_images = [
      MedicalImage(f'Sample-{i}', images={
        'Input': self.features[i],
        'Targets': self.targets[i],
        'Model-Output': pred[i]})
      for i in range(self.size)]

    dg = DrGordon(medical_images)
    dg.slice_view.set('vmin', auto_refresh=False)
    dg.slice_view.set('vmax', auto_refresh=False)
    dg.show()

  def snapshot(self, model):
    from tframe import Predictor
    import os
    import matplotlib.pyplot as plt

    assert isinstance(model, Predictor)

    # (1) Get denoised image (shape=[1, D, H, W, 1])
    denoised_images = model.predict(self)

    # (2) Get metrics
    val_dict = model.validate_model(self)

    # (3) Save image
    metric_str = '-'.join([f'{k}{v:.2f}' for k, v in val_dict.items()])
    fn = f'Iter{model.counter}-{metric_str}.png'
    plt.imsave(os.path.join(model.agent.ckpt_dir, fn),
               denoised_images[0, ..., 0])

  # endregion: Validation and snapshot
