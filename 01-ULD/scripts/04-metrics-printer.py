from utils.metrics_calc import get_metrics
from xomics.data_io.npy_reader import load_data

dose_tags = [
  'Full',
  '1-2',
  '1-4',
  '1-10',
  '1-20',
  '1-50',
  '1-100',
]


def output_metrics(arr1, arr2, metrics: list):
  result = get_metrics(arr1, arr2, metrics)
  for metric, value in result.items():
    print(f'{metric}: {value}')

  return result


if __name__ == '__main__':
  dirpath = '../../data/01-ULD/'
  shape = [1, 608, 440, 440, 1]
  full = load_data(dirpath, 6, dose_tags[0], shape=shape)[0, ..., 0]
  print(full.shape)

  metr = ['nrmse', 'SSIM', 'psnr', 'pw_rmse']
  for i in range(len(dose_tags[1:])):
    print(f'{dose_tags[1:][i]} dose:')
    img = load_data(dirpath, 6, dose_tags[1:][i], shape=shape)[0, ..., 0]
    output_metrics(full, img, metr)

