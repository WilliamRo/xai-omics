from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from pictor.xomics.omix import Omix
from roma import finder, console

import os
import SimpleITK as sitk



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
data_dir = r'D:\data\BAMIA\CT'
target = 1

data_dir = os.path.join(data_dir, ['BA', 'MIA'][target])
mask_dir = os.path.join(data_dir, 'labels')
mask_files = finder.walk(mask_dir, pattern='*.nii')

# -----------------------------------------------------------------------------
# Analyze HU range
# -----------------------------------------------------------------------------
N, M = len(mask_files), 0

for i, mask_file in enumerate(mask_files):
  console.print_progress(i, N)

  file_name = os.path.basename(mask_file)
  pid, iid = file_name.split('.')[0].split('-')[:2]

  image_name = f'{pid}-{iid}-CT.nii.gz'
  image_path = os.path.join(mask_dir, image_name)
  assert os.path.exists(image_path), f'`{image_path}` does not exist.'

  img = sitk.ReadImage(image_path)

  array = sitk.GetArrayFromImage(img)
  hu_min, hu_max = array.min(), array.max()

  console.show_status(f'{pid}-{iid}: [{hu_min}, {hu_max}]')

  M += 1
