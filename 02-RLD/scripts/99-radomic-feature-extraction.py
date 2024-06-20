import numpy as np
import os
import csv
import sys
import SimpleITK as sitk

from radiomics import featureextractor
from time import time

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
data_path = os.path.join(root_path, 'out_data')
# os.chdir(data_path)

data_dir = ['30s_gated', '60s_gated', '90s_gated', '120s_gated']
time_types = ['30G', '60G-1', '90G', '120G']
models_dir = ['raw_data', 'gaussian', 'unet', 'gan']
raw_dir = 'raw_data'
target_dir = '240s_gated_raw'
target_name = '240G'
mask_dir = 'CT_seg'

region_dict = {
  'lung': [10, 11, 12, 13, 14],
  'liver': [5]
}

# with open('test_id', 'r') as f:
#   pids = f.read().splitlines()
pids = ['10361']
pids = [f'YHP000{_}' for _ in pids]


def catch_features(image_path, mask_path):
  settings = {}
  settings['binWidth'] = 25  # 5
  settings['sigma'] = [3, 5]
  settings['Interpolator'] = sitk.sitkBSpline
  settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
  settings['voxelArrayShift'] = 1000  # 300
  settings['normalize'] = True
  settings['normalizeScale'] = 100
  settings['verbose'] = True

  extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

  extractor.enableImageTypeByName('LoG')
  extractor.enableImageTypeByName('Wavelet')

  extractor.enableAllFeatures()

  # extractor.enableFeaturesByName(
  #   firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile',
  #               '90Percentile', 'Maximum', 'Mean', 'Median',
  #               'InterquartileRange', 'Range', 'MeanAbsoluteDeviation',
  #               'RobustMeanAbsoluteDeviation', 'RootMeanSquared',
  #               'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
  # extractor.enableFeaturesByName(
  #   shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio',
  #          'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion',
  #          'Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn',
  #          'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
  #          'LeastAxisLength', 'Elongation', 'Flatness'])

  feature_cur, feature_name = [], []
  result = extractor.execute(image_path, mask_path, label=1)
  for key, value in result.items():
    feature_name.append(key)
    feature_cur.append(value)

  name = np.array(feature_name[37:])

  for i in range(len(feature_cur[37:])):
    feature_cur[i + 37] = float(feature_cur[i + 37])

  return np.array(feature_cur[37:]), name


def mask2onehot(seg, labels: list):
  onehot = np.zeros_like(seg, dtype=bool)
  onehot[np.isin(seg, labels)] = True
  return onehot.astype(np.short)


def get_region(mask_path, region):
  import SimpleITK as sitk
  mask = sitk.ReadImage(mask_path)
  region_id = region_dict[region]

  mask_arr = sitk.GetArrayViewFromImage(mask)
  mask_arr = mask2onehot(mask_arr, region_id)

  new_mask = sitk.GetImageFromArray(mask_arr)
  new_mask.CopyInformation(mask)
  return new_mask


def adjust_data(image_path, mask):
  import SimpleITK as sitk
  img = sitk.ReadImage(image_path)
  img.SetOrigin(mask.GetOrigin())
  img.SetSpacing(mask.GetSpacing())
  img.SetDirection(mask.GetDirection())
  return img



if __name__ == '__main__':
  from roma import console

  data_dir = r'D:\data\ULD-radomics\test-1'
  image_path = os.path.join(data_dir, '0-YHP00010361-30G.nii.gz')
  mask_path = os.path.join(data_dir, r'0-YHP00010361-mask.nii.gz')

  region = ['lung', 'liver'][1]

  console.show_status('Getting region mask ...')
  mask = get_region(mask_path, region)
  console.show_status('Adjusting data ...')
  img = adjust_data(image_path, mask)

  console.show_status('Catching features ...')
  feature_cur, feature_name = catch_features(img, mask)

  print()
