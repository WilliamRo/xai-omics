import numpy as np

from tframe import DataSet



class ULDSet(DataSet):

  @property
  def eval_set(self):
    from uld_core import th
    s = th.eval_window_size

    # TODO: Generate dataset for evaluation
    x = np.zeros(shape=[self.size, s, s, s, 1])

    return ULDSet(x, x)


  def gen_random_window(self, batch_size):
    from uld_core import th
    # TODO: Randomly sample [S, S, S] pair from features and targets

    # self.features/targets.shape = [N, S, H, W, 1]
    s = th.window_size
    x = np.zeros(shape=[batch_size, s, s, s, 1])
    data_batch = DataSet(x, x)

    return data_batch


  def gen_batches(self, batch_size, shuffle=False, is_training=False):

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
