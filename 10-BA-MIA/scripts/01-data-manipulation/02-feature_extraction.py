from pictor.xomics.radiomics.feature_extractor import RadiomicFeatureExtractor
from pictor.xomics.omix import Omix
from roma import finder, console

import os



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
settings = {
  'binWidth': 25,
  # 'binWidth': 1,
  # 'additionalInfo': False,
  'normalize': True,
  'resampledPixelSpacing': [2, 2, 2],
}

filters = ['LoG', 'Wavelet']

data_dir = r'D:\data\BAMIA\CT'
feature_dir = os.path.join(data_dir, 'rad_features_pool')
target = 1
data_dir = os.path.join(data_dir, ['BA', 'MIA'][target])

mask_dir = os.path.join(data_dir, 'labels')

mask_files = finder.walk(mask_dir, pattern='*.nii')

# -----------------------------------------------------------------------------
# Extract features
# -----------------------------------------------------------------------------
N, M = len(mask_files), 0

fe = RadiomicFeatureExtractor(settings=settings, filters=filters)
for i, mask_file in enumerate(mask_files):
  console.print_progress(i, N)

  file_name = os.path.basename(mask_file)
  pid, iid = file_name.split('.')[0].split('-')[:2]

  image_name = f'{pid}-{iid}-CT.nii.gz'
  image_path = os.path.join(mask_dir, image_name)
  assert os.path.exists(image_path), f'`{image_path}` does not exist.'

  omix_fn = f'{target}-{pid}-{iid}.omix'
  save_path = os.path.join(feature_dir, omix_fn)

  # OVERWRITE
  # if os.path.exists(save_path): continue

  x, labels = fe.extract_features_from_nii(
    image_path, mask_file, mask_labels=1, verbose=0)

  omix = Omix(x.reshape(1, -1), targets=[target], feature_labels=labels,
              sample_labels=[f'{pid}-{iid}'], target_labels=['BA', 'MIA'],
              data_name=f'{pid}')
  omix.save(save_path)

  M += 1

console.show_status(f'Extracted {M}/{N} files.')