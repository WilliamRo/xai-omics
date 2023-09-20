from uld.uld_agent import ULDAgent, ULDSet, DataSet
from roma import console
from xomics.data_io.uld_reader import UldReader

import numpy as np



def load_data():
  from uld_core import th

  datasets = ULDAgent.load(th.data_dir, th.val_size, th.test_size)

  for ds in datasets:
    ds.data_fetcher = None
    if 'chip' in th.developer_code:
      ds.batch_preprocessor = batch_preprocessor

  return datasets


def batch_preprocessor(data_batch: DataSet, is_training: bool):
  from uld_core import th
  if not th.train: return data_batch

  features, targets = data_batch.features, data_batch.targets
  N = features.shape[0]
  T = th.thickness

  xs, ys = [], []
  if is_training:
    for _ in range(th.uld_batch_size):
      pi = np.random.randint(0, N)
      si = np.random.randint(0, features.shape[1] - T)
      xs.append(features[pi][si:si+T])
      ys.append(targets[pi][si:si+T])
  else:
    for low, full in zip(features, targets):
      # In this way, if T = 30, 100 slices will be split into [30, 30, 30, 10]
      for i in range(int(np.ceil(low.shape[0] / T))):
        xs.append(low[i*T:(i+1)*T])
        ys.append(full[i*T:(i+1)*T])

  data_batch.features = np.stack(xs, axis=0)
  data_batch.targets = np.stack(ys, axis=0)

  return data_batch
