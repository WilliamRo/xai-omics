from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor

import os



# -----------------------------------------------------------------------------
# (0) Configurations
# -----------------------------------------------------------------------------
data_dir = r'P:\ibsi\ibsi_1_ct_radiomics_phantom\nifti'

image_path = os.path.join(data_dir, r'image\phantom.nii.gz')
mask_path = os.path.join(data_dir, r'mask\mask.nii.gz')


# -----------------------------------------------------------------------------
# (1) Extract features
# -----------------------------------------------------------------------------
extractor = RadiomicFeatureExtractor()

# mask = extractor.get_mask_from_nii( mask_path, 1, plot=1, plot_settings={'step_size': 3})

feature_dict = extractor.extract_features_from_nii(
  image_path, mask_path, mask_labels=1, verbose=1, return_fmt='dict')

# Print the extracted features
for key, value in feature_dict.items(): print(f'{key}: {value}')