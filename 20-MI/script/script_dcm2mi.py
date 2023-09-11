import pydicom
import os
import numpy as np

from xomics import MedicalImage



if __name__ == "__main__":
  dir = r'E:\PET\Brain Seg\data'
  patient_ids = os.listdir(dir)
  patient_ids = [id for id in patient_ids if id != 'mi']

  for id in patient_ids:
    patient_dir = os.path.join(dir, id)
    types = os.listdir(patient_dir)
    save_dir = os.path.join(dir, 'mi', id + '.mi')

    mi: MedicalImage = MedicalImage()

    for i, type in enumerate(types):
      dcm_dir = os.path.join(patient_dir, type)
      dcm_file = os.listdir(dcm_dir)

      dcms = [pydicom.dcmread(os.path.join(dcm_dir, file))
              for file in dcm_file]
      dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
      data = [d.pixel_array for d in dcms]

      # data = [pydicom.dcmread(os.path.join(dcm_dir, file)).pixel_array
      #         for file in dcm_file]
      # data = pydicom.dcmread(os.path.join(dcm_dir, dcm_file[0]))

      mi.images[f'ct-{i}'] = np.array(data, dtype=np.float32)
      mi.key = id
      break

    mi.save(save_dir)


