from xomics.objects import MedicalImage
from tqdm import tqdm

import os
import nibabel as nib
import numpy as np


if __name__ == '__main__':
  image_dir = r'../../data/02-PET-CT-Y1/sg_raw'
  mask_dir = r'../../data/02-PET-CT-Y1/sg_ROI1.1'
  save_dir = r'../../data/02-PET-CT-Y1/mi'

  ct_name = 'ct.nii'
  pet_name = 'pet-resample.nii'

  patient_id = os.listdir(mask_dir)
  for p in tqdm(patient_id):
    mask_path = os.path.join(mask_dir, p)
    file = os.listdir(mask_path)

    mask_file = os.path.join(mask_path, file[0])
    ct_file = os.path.join(image_dir, p, ct_name)
    pet_file = os.path.join(image_dir, p, pet_name)

    mask = np.transpose(
      np.array(nib.load(mask_file).get_fdata()), axes=(2, 1, 0))
    ct = np.transpose(
      np.array(nib.load(ct_file).get_fdata()), axes=(2, 1, 0))
    pet = np.transpose(
      np.array(nib.load(pet_file).get_fdata()), axes=(2, 1, 0))

    mask = mask.astype(np.int8)
    ct = ct.astype(np.float16)
    pet = (pet / 2**6).astype(np.float16)

    assert not np.any(np.isinf(ct)) and not np.any(np.isinf(pet))

    assert mask.shape == ct.shape == pet.shape

    image = {'ct': ct, 'pet': pet}
    label = {'label-0': mask}

    mi = MedicalImage(images=image, labels=label, key=p)
    save_file = os.path.join(save_dir, p + '.mi')
    mi.save(save_file)










