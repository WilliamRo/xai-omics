from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor

import os



# -----------------------------------------------------------------------------
# (0) Configurations
# -----------------------------------------------------------------------------
data_dir = r'D:\data\ULD-radomics\test-1'

image_path = os.path.join(data_dir, '0-YHP00010361-30G.nii.gz')
mask_path = os.path.join(data_dir, r'0-YHP00010361-mask.nii.gz')

region = ['lung', 'liver'][1]

region_dict = {
  'lung': [10, 11, 12, 13, 14],
  'liver': [5]
}
labels = region_dict[region]
# -----------------------------------------------------------------------------
# (1) Extract features
# -----------------------------------------------------------------------------
extractor = RadiomicFeatureExtractor()

feature_dict = extractor.extract_features_from_nii(
  image_path, mask_path, mask_labels=labels, verbose=1, return_fmt='dict')

# Print the extracted features
for key, value in feature_dict.items(): print(f'{key}: {value}')
