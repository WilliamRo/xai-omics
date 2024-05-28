from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from roma import finder, console

import os



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
data_dir = r'D:\data\BAMIA\CT'
data_dir = os.path.join(data_dir, ['BA', 'MIA'][1])

# -----------------------------------------------------------------------------
# Converting
# -----------------------------------------------------------------------------
mask_dir = os.path.join(data_dir, 'labels')

mask_files = finder.walk(mask_dir, pattern='*.nii')
N, M = len(mask_files), 0

fe = RadiomicFeatureExtractor()
for i, mask_file in enumerate(mask_files):
  console.print_progress(i, N)

  file_name = os.path.basename(mask_file)

  pid, iid = file_name.split('.')[0].split('-')[:2]

  dcm_dir = os.path.join(data_dir, f'{pid}/{iid}')

  assert os.path.exists(dcm_dir)

  save_dir = mask_dir
  file_name = f'{pid}-{iid}.nii.gz'

  new_name = f'{pid}-{iid}-CT.nii.gz'

  if os.path.exists(os.path.join(save_dir, file_name)):
    os.rename(os.path.join(save_dir, file_name),
              os.path.join(save_dir, new_name))
    M += 1

console.show_status(f'Converted {M}/{N} files.')
