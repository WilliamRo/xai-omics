from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from pictor.xomics.radiomics import rad_plotters
from roma import console

import os
import matplotlib.pyplot as plt



# -----------------------------------------------------------------------------
# (0) Configurations
# -----------------------------------------------------------------------------
data_dir = r'D:\data\ULD-radomics\test-1'

image_path = os.path.join(data_dir, '0-YHP00010361-30G.nii.gz')
mask_path = os.path.join(data_dir, r'0-YHP00010361-mask.nii.gz')

region = ['lung', 'liver'][0]

region_dict = {
  'lung': [10, 11, 12, 13, 14],
  'liver': [5]
}
labels = region_dict[region]
# -----------------------------------------------------------------------------
# (1) Extract mask
# -----------------------------------------------------------------------------
rfe = RadiomicFeatureExtractor()

console.show_status('Reading mask ...')
mask = rfe.get_mask_from_nii(mask_path, labels, return_array=True)

console.show_info(f'Mask shape: {mask.shape}')

# -----------------------------------------------------------------------------
# (2) Show in explorer
# -----------------------------------------------------------------------------
console.show_status('Plotting Voxels ...')

rad_plotters.plot_3D_mask(mask, step_size=5)





