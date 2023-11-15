import pydicom
import os
import numpy as np

from xomics import MedicalImage
from xomics.data_io.utils.preprocess import calc_SUV
from tqdm import tqdm



if __name__ == "__main__":
  dir = r'E:\xai-omics\data\04-Brain-CT-PET'
  raw_dir = os.path.join(dir, 'raw data')
  mi_dir = os.path.join(dir, 'mi')
  patient_ids = os.listdir(raw_dir)
  patient_ids = [id for id in patient_ids if id != 'mi']

  for id in tqdm(patient_ids):
    patient_dir = os.path.join(raw_dir, id)
    types = os.listdir(patient_dir)
    save_dir = os.path.join(mi_dir, id + '.mi')

    mi: MedicalImage = MedicalImage()

    for i, type in enumerate(types):
      dcm_dir = os.path.join(patient_dir, type)
      dcm_file = os.listdir(dcm_dir)

      dcms = [pydicom.dcmread(os.path.join(dcm_dir, file))
              for file in dcm_file]
      dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
      data = np.array([d.pixel_array for d in dcms], dtype=np.uint16)
      if data[0].shape[1] == 512: continue

      # calculate the suv
      dcm = dcms[0]
      ST = dcm.SeriesTime
      AT = dcm.AcquisitionTime
      PW = dcm.PatientWeight
      RIS = dcm.RadiopharmaceuticalInformationSequence[0]
      RST = str(RIS['RadiopharmaceuticalStartTime'].value)
      RTD = str(RIS['RadionuclideTotalDose'].value)
      RHL = str(RIS['RadionuclideHalfLife'].value)
      RS = dcm.RescaleSlope
      RI = dcm.RescaleIntercept
      dcm_tag = {
        'ST': ST,
        'AT': AT,
        'PW': PW,
        'RST': RST,
        'RTD': RTD,
        'RHL': RHL,
        'RS': RS,
        'RI': RI
      }
      data = calc_SUV(data, tags=dcm_tag, norm=False)

      mi.images[f'pet'] = np.array(data, dtype=np.uint16)
      mi.key = id

    mi.save(save_dir)


