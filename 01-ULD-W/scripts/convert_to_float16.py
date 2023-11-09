from roma import finder
from roma import console

import numpy as np
import os



src_dir = '../../data/01-ULD/'
tgt_dir = '../../data/01-ULD/'

for sub_name in finder.walk(src_dir, return_basename=True):
  console.show_status(f'Converting {sub_name} ...')
  sub_dir_src = os.path.join(src_dir, sub_name)
  sub_dir_tgt = os.path.join(tgt_dir, sub_name)

  file_names = finder.walk(
    sub_dir_src, pattern='subject*_[1F]*.npy', return_basename=True)
  for i, fn in enumerate(file_names):
    console.print_progress(i, len(file_names))

    src_path = os.path.join(sub_dir_src, fn)
    tgt_path = os.path.join(sub_dir_tgt, fn)
    data = np.load(src_path)
    data2 = data.astype(np.uint32)
    # np.save(tgt_path, data.astype(np.float16))

  console.show_status(f'{len(file_names)} files in {sub_name} converted.')





