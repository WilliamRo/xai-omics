import Script.script_get_data_from_nii
from xomics.gui.dr_gordon import DrGordon
from xomics import MedicalImage

import numpy as np
import os



def load_npy_file(file_path):
  return np.load(file_path, allow_pickle=True).tolist()


input_dir = r'..\data\npy_data'
ct_file = '2hcxia_ct.npy'
pet_file = '2hcxia_pet.npy'
label_file = '2hcxia_mask.npy'

# Loading data
# ct and label have the same shape
# [patient, slice, H, W]
ct = load_npy_file(os.path.join(input_dir, ct_file))
pet = load_npy_file(os.path.join(input_dir, pet_file))
label = load_npy_file(os.path.join(input_dir, label_file))

# Normalization
ct = np.squeeze(ct) / np.max(ct)
pet = np.squeeze(pet) / np.max(pet)
label = np.squeeze(label) / np.max(label)


pet_file = r'..\..\data\PET\2hcxia\resample-pet\PET-512.nii'
pet = Script.script_get_data_from_nii.read_nii_with_medpy(pet_file)[::-1]
pet = np.squeeze(pet) / np.max(pet)


mi_1 = MedicalImage('Patient-1', {'CT': ct, 'PET': pet},
                    {'Label-1': label})
# mi_2 = MedicalImage('Patient-2', {'CT': ct[3:]}, {'Label-1': label[3:]})


# Visualization
dg = DrGordon([mi_1])
dg.show()
