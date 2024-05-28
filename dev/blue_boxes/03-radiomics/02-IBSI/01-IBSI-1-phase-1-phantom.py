from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor

import os



# -----------------------------------------------------------------------------
# (0) Configurations
# -----------------------------------------------------------------------------
data_dir = r'P:\ibsi\ibsi_1_digital_phantom\nifti'

image_path = os.path.join(data_dir, r'image\phantom.nii.gz')
mask_path = os.path.join(data_dir, r'mask\mask.nii.gz')

settings = {
  'sigma': [3, 5],
}

filters = [
  'LoG',
  # 'Square',
  'Wavelet',
]
# -----------------------------------------------------------------------------
# (1) Extract features
# -----------------------------------------------------------------------------
extractor = RadiomicFeatureExtractor(filters=filters, settings=settings)

feature_dict = extractor.extract_features_from_nii(
  image_path, mask_path, mask_labels=1, verbose=1, return_fmt='raw')

# Print the extracted features
# for key, value in feature_dict.items(): print(f'{}: {value}')
