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
  image_dir = r'E:\xai-omics\data\02-PET-CT-Y1\nii\ecs_suv_norm'
  mask_dir = r'E:\xai-omics\data\02-PET-CT-Y1\results\25-ESO\nii\1126_(25-ESO)_unet(4-5-4-1-relu-mp)_Sc_11_raw_256'

  save_dir = r'../../../data/02-PET-CT-Y1/features'
  save_file_name = r'data_gt_256_suv_ns_t.xlsx'

  feature_data = []
  pids = []
  patient_id = os.listdir(image_dir)

  for p in tqdm(patient_id, desc='Processing ...'):
    if '106' in p: continue
    ct_path = os.path.join(image_dir, p, 'ct.nii')
    pet_path = os.path.join(image_dir, p, 'pet.nii')

    mask_file = r'mask.nii'
    mask_path = os.path.join(mask_dir, p, mask_file)

    ct_curdata, ct_name = catch_features(ct_path, mask_path)
    pet_curdata, pet_name = catch_features(pet_path, mask_path)

    assert np.array_equal(ct_name, pet_name)
    name = ct_name

    feature_data.append(np.concatenate((ct_curdata, pet_curdata)))
    pids.append(p)

  feature_names = [n + '_ct' for n in name] + [n + '_pet' for n in name]

  workbook = openpyxl.Workbook()
  sheet = workbook.active
  sheet.append(['PIDS'] + feature_names)

  for p, d in zip(pids, feature_data):
    sheet.append([p] + d.tolist())

  workbook.save(os.path.join(save_dir, save_file_name))
