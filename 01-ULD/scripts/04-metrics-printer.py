import numpy as np

from utils.metrics_calc import get_metrics
from xomics.data_io.uld_reader import UldReader

dose_tags = [
  ['Full'],
  ['1-2'],
  ['1-4'],
  ['1-10'],
  ['1-20'],
  ['1-50'],
  ['1-100'],
]


def output_metrics(arr1, arr2, metrics: list, **kwargs):
  result = get_metrics(arr1, arr2, metrics, **kwargs)
  for metric, value in result.items():
    print(f'{metric}: {value: .5f}')

  return result


if __name__ == '__main__':
  dirpath = '../../data/01-ULD/'
  shape = [1, 608, 440, 440, 1]
  subjects = [2]

  reader = UldReader(dirpath)
  imgs = reader.load_data(subjects, dose_tags, methods='type',
                          shape=shape, raw=True)
  full = imgs['Full'][0][0, ..., 0]

  metr = ['nrmse', 'SSIM', 'psnr']
  for k, v in imgs.items():
    if k == 'Full': continue
    print(f'{k} dose:')
    norm = np.max(v[0][0, ..., 0])
    output_metrics(full/norm, v[0][0, ..., 0]/norm, metr, data_range=1.0)

