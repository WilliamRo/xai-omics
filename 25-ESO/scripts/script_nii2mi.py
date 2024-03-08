from xomics.objects import MedicalImage
from tqdm import tqdm

import os
import nibabel as nib
import numpy as np


if __name__ == '__main__':
  image_dir = r'../../data/02-PET-CT-Y1/sg_raw'

  mask_lesion_dir = r'../../data/02-PET-CT-Y1/sg_Roi_ecs'
  mask_region_dir = r'../../data/02-PET-CT-Y1/sg_Roi_es'

  save_dir = r'../../data/02-PET-CT-Y1/mi/both'

  ct_name = 'ct.nii'
  pet_name = 'pet-resample.nii'

  patient_id = os.listdir(image_dir)
  for p in tqdm(patient_id, desc='Processing'):
    # Parameter setting
    image, label = {}, {}

    # Get CT data and PET data
    ct_file = os.path.join(image_dir, p, ct_name)
    pet_file = os.path.join(image_dir, p, pet_name)

    ct = np.transpose(
      np.array(nib.load(ct_file).get_fdata()), axes=(2, 1, 0))
    pet = np.transpose(
      np.array(nib.load(pet_file).get_fdata()), axes=(2, 1, 0))

    ct = ct.astype(np.float16)
    pet = (pet / 2**6).astype(np.float16)

    assert not np.any(np.isinf(ct)) and not np.any(np.isinf(pet))
    assert ct.shape == pet.shape

    image = {'ct': ct, 'pet': pet}

    # Get mask data
    mask_names = ['lesion', 'region']
    mask_dir = [mask_lesion_dir, mask_region_dir]
    for i, n in enumerate(mask_names):
      mask_path = os.path.join(mask_dir[i], p)
      if not os.path.exists(mask_path): continue
      file = os.listdir(mask_path)[0]

      mask_file = os.path.join(mask_path, file)

      mask = np.transpose(
        np.array(nib.load(mask_file).get_fdata()), axes=(2, 1, 0))
      mask = mask.astype(np.int8)
      label[n] = mask

      assert mask.shape == ct.shape

    mi = MedicalImage(images=image, labels=label, key=p)
    save_file = os.path.join(save_dir, p + '.mi')
    print(f"PID: {p} --- Number of Labels: {len(label)}")
    mi.save(save_file)
