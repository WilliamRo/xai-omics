import six
import os  # needed navigate the system to get the input data
import numpy as np
import pickle
import radiomics
import SimpleITK as sitk
import openpyxl

from radiomics import featureextractor  # This module is used for interaction with pyradiomics
from tqdm import tqdm



def catch_features(image_path, mask_path):
  settings = {}
  settings['binWidth'] = 25  # 5
  settings['sigma'] = [3, 5]
  settings['Interpolator'] = sitk.sitkBSpline
  settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
  settings['voxelArrayShift'] = 1000  # 300
  settings['normalize'] = True
  settings['normalizeScale'] = 100
  extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
  # print('Extraction parameters:\n\t', extractor.settings)

  extractor.enableImageTypeByName('LoG')
  extractor.enableImageTypeByName('Wavelet')
  extractor.enableAllFeatures()
  extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation', 'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
  extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'Sphericity', 'SphericalDisproportion','Maximum3DDiameter','Maximum2DDiameterSlice','Maximum2DDiameterColumn','Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])
  # 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
  # print('Enabled filters:\n\t', extractor.enabledImagetypes)

  feature_cur, feature_name = [], []
  result = extractor.execute(image_path, mask_path, label=1)
  for key, value in six.iteritems(result):
    feature_name.append(key)
    feature_cur.append(value)

  name = np.array(feature_name[37:])

  for i in range(len(feature_cur[37:])):
    feature_cur[i+37] = float(feature_cur[i+37])

  return feature_cur[37:], name



if __name__ == '__main__':
  '''
  Extract radiomics features by pyradiomics
  '''
  # Input
  image_dir = r'E:\xai-omics\data\31-Prostate\nii_128'
  mask_dir = image_dir

  # Output
  save_dir = r'E:\xai-omics\data\31-Prostate\results'
  save_file_name = r'features_radiomics_395_aorta.xlsx'

  # Parameter
  feature_data = []
  patient_id = os.listdir(image_dir)

  for p in tqdm(patient_id, desc='Processing ...'):
    ct_path = os.path.join(image_dir, p, 'ct.nii.gz')
    pet_path = os.path.join(image_dir, p, 'pet.nii.gz')
    mask_path = os.path.join(mask_dir, p, 'mask.nii.gz')

    if os.path.exists(ct_path) and os.path.exists(pet_path) and os.path.exists(mask_path):
      pass
    else:
      print(f'{p} missing files')
      continue

    ct_curdata, ct_name = catch_features(ct_path, mask_path)
    pet_curdata, pet_name = catch_features(pet_path, mask_path)

    assert np.array_equal(ct_name, pet_name)
    name = ct_name

    feature_data.append([p] + list(np.concatenate((ct_curdata, pet_curdata))))
    # feature_data.append(pet_curdata)

  feature_names = [n + '_ct' for n in name] + [n + '_pet' for n in name]

  workbook = openpyxl.Workbook()
  sheet = workbook.active
  sheet.append(['PIDS'] + list(feature_names))

  for d in feature_data:
    sheet.append(d)

  workbook.save(os.path.join(save_dir, save_file_name))
  print(f'{save_file_name} saved successfully!')
