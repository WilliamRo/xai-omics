import os.path

from xomics.data_io.uld_reader import UldReader

import nibabel as nib

def nii_rd(path, subs):
  arr = []
  for i in subs:
    nii_data = nib.load(os.path.join(path, f'Anonymous_{i}.nii.gz'))
    data = nii_data.get_fdata()
    arr.append(data)
  return arr


if __name__ == '__main__':
  testpath = '../../data/01-ULD/testset/'
  subs = [1, 50]
  reader = UldReader.load_as_npy_data(testpath, subs,
                                      ('Anonymous_', '.nii.gz'), raw=True)
  img_sitk = reader.data
  img_nii = nii_rd(testpath, subs)

  print()