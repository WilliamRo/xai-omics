from tframe import DataSet

import random
import numpy as np



class PROSet(DataSet):

  def report(self):
    pass


  def _check_data(self):
    pass


  def test_model(self, model):
    threshold_bias = 0.15
    results = model.predict(self)
    pred = np.argmax(results, axis=1)
    gt = self.targets[:, 1]
    raw_accuracy = np.sum(pred == gt) / len(pred)

    effect_list = []
    for conf in results:
      if np.abs(conf[0] - conf[1]) <= threshold_bias * 2: effect_list.append(0)
      else: effect_list.append(1)

    effect_list = np.array(effect_list)

    effect_pred = pred[effect_list == 1]
    effect_gt = gt[effect_list == 1]

    effect_accuracy = np.sum(effect_pred == effect_gt) / len(effect_pred)

    uncertain_pid = self.data_dict['pids'][effect_list == 0]
    misclassified_pid = self.data_dict['pids'][pred != gt]
    print(f'Raw Acc: {round(raw_accuracy, 4)}\t\t '
          f'Effect Acc: {round(effect_accuracy, 4)}\t\t')
    print(f'Uncertainty: {uncertain_pid}')
    print(f'Misclassified: {misclassified_pid}')

    print()


  def split(self, *sizes, names=None, over_classes=False, random=False):
    """If over_classes is True, sizes are weights for each group.
    e.g.,
       train_set, val_set = train_set.split(9, 1, over_classes=True)
    """
    # Sanity check
    if len(sizes) == 0: raise ValueError('!! split sizes not specified')
    elif len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
      # in case this method is used like split([-1, 100, 100])
      #   instead of split(-1, 100, 100)
      sizes = sizes[0]
    if names is not None:
      if not isinstance(names, (tuple, list)):
        raise TypeError('!! names must be a tuple or list of strings')
      if len(names) != len(sizes):
        raise ValueError('!! length of name list and sizes list does not match')
    # Check sizes
    sizes, auto_index, size_accumulator = list(sizes), -1, 0
    for i, size in enumerate(sizes):
      if size is None or size < 0:
        if auto_index < 0:
          auto_index = i
          continue
        else: raise ValueError(
          '!! only one split size can be calculated automatically')
      if not isinstance(size, int) and size < 0:
        raise ValueError('!! size must be a non-negative integer')
      size_accumulator += size

    # Get total size according to over_classes flag
    total_size = self.size
    if over_classes:
      # Auto-size is forbidden when `over_classes` option is on
      if auto_index >= 0: raise AssertionError(
        '!! Auto-size is not allowed when `over_classes` is True')

      # ----------------------------------------------------------------- x
      # TODO 1
      # # Find sample numbers for each class
      # sample_nums = [len(g) for g in self.groups]
      #
      # for n in sample_nums: assert n == sample_nums[0]
      # total_size = sample_nums[0]
      # ----------------------------------------------------------------- x

    # Calculate size automatically if necessary
    if auto_index >= 0:
      sizes[auto_index] = total_size - size_accumulator
      if sizes[auto_index] < 0: raise ValueError(
        '!! negative value appears when calculating size automatically')
    elif not over_classes and size_accumulator != total_size: raise ValueError(
      '!! total size does not match size of the data set to split')

    # Split data set
    data_sets, cursor = (), 0
    indices_pool = set(range(self.size))
    if over_classes:
      group_pool = [g[:] for g in self.groups]
      if random: group_pool = [np.random.permutation(g) for g in group_pool]
      # ----------------------------------------------------------------- x
      # TODO 2
      # group_pool = [set(range(total_size)) for _ in self.groups]
      # ----------------------------------------------------------------- x
    for i, size in enumerate(sizes):
      if size == 0: continue
      # Generate indices
      if not over_classes:
        if not random:
          indices = slice(cursor, cursor + size)
        else:
          indices = np.random.choice(list(indices_pool), size, replace=False)
          indices_pool -= set(indices)
      else:
        # Calculate proportion of i-th sub-dataset
        p = size / size_accumulator
        # Wrap remains as last dataset
        if i == len(sizes) - 1:
          indices = np.concatenate(group_pool).astype(np.int)
        else:
          indices = []
          for j, g in enumerate(group_pool):
            n = round(p * len(self.groups[j]))
            indices.extend(g[:n])
            group_pool[j] = g[n:]

          # ----------------------------------------------------------------- x
          # TODO 3
          # if not random:
          #   for g in self.groups:
          #     indices += g[slice(cursor, cursor + size)]
          # else:
          #   for j, g in enumerate(group_pool):
          #     idcs = np.random.choice(list(g), size, replace=False)
          #     group_pool[j] = g - set(idcs)
          #     for idx in idcs: indices.append(self.groups[j][idx])
          # ----------------------------------------------------------------- x

      # Get subset
      data_set = self[indices]
      if names is not None: data_set.name = names[i]
      data_sets += (data_set,)
      cursor += size

    return data_sets



if __name__ == '__main__':
  pass

