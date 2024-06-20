import numpy as np
import os

from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from roma import console



# -----------------------------------------------------------------------------
# (0) Configurations
# -----------------------------------------------------------------------------
data_dir = r'P:\ibsi\ibsi_1_digital_phantom\nifti'

image_path = os.path.join(data_dir, r'image\phantom.nii.gz')
mask_path = os.path.join(data_dir, r'mask\mask.nii.gz')

settings = {
  'sigma': [3, 5],
  'binWidth': 5,
  'resampledPixelSpacing': [1, 1, 1],
}

filters = [
  'LoG',
  'Wavelet',
  # 'Square',
  # 'SquareRoot',
  # 'Logarithm',
  # 'Exponential',
]
# -----------------------------------------------------------------------------
# (1) Extract features
# -----------------------------------------------------------------------------
extractor = RadiomicFeatureExtractor(filters=filters, settings=settings)

feature_dict = extractor.extract_features_from_nii(
  image_path, mask_path, mask_labels=1, verbose=1, return_fmt='dict')

keys = list(feature_dict.keys())
filter_keys = np.unique([k.split('_')[0] for k in keys])
for fk in filter_keys:
  fk_keys = [k for k in keys if k.startswith(fk)]
  console.show_info(f'Filter: {fk} ({len(fk_keys)} features)')
  group_keys = np.unique([k.split('_')[1] for k in fk_keys])
  for gk in group_keys:
    gk_keys = [k for k in fk_keys if k.startswith(f'{fk}_{gk}')]
    console.supplement(f'{gk}: {len(gk_keys)} features', level=2)


# print(filter_keys)

# print(feature_dict)

# Print the extracted features
# for key, value in feature_dict.items(): print(f'{}: {value}')
