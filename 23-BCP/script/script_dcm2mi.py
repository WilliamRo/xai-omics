import pydicom
import os
import numpy as np
import pandas as pd

from xomics import MedicalImage
from xomics.data_io.utils.preprocess import calc_SUV
from tqdm import tqdm



if __name__ == "__main__":
  dir = r'E:\xai-omics\data\04-Brain-CT-PET'
  raw_dir = os.path.join(dir, 'raw data')
  mi_dir = os.path.join(dir, 'mi')
  label_path = os.path.join(dir, 'p3_cit_标签1.xlsx')
  patient_ids = os.listdir(raw_dir)
  patient_ids = [id for id in patient_ids if id != 'mi']

  # Get the labels of patients

  df = pd.read_excel(label_path)
  PIDs, labels = df.values[:, 1].astype(np.int), df.values[:, 3]

  c_t_e= {'左': 'left', '右': 'right', '双侧': 'both', '正常': 'normal'}
  labels = np.vectorize(c_t_e.get)(labels)

  for id in tqdm(patient_ids):
    if '0174' in id or '0347' in id:
      continue

    patient_dir = os.path.join(raw_dir, id)

    dcm_dir = os.path.join(patient_dir, 'pet')
    dcm_file = os.listdir(dcm_dir)

    dcms = [pydicom.dcmread(os.path.join(dcm_dir, file))
            for file in dcm_file]
    dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
    data = np.array([d.pixel_array for d in dcms], dtype=np.uint16)

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

    mi: MedicalImage = MedicalImage()
    mi.images[f'pet'] = np.array(data, dtype=np.uint16)

    pid, pname = id.split('_')
    label = labels[np.where(PIDs == int(pid))][0]
    mi.key = f'{pid}-{pname}-{label}'
    save_dir = os.path.join(mi_dir, mi.key + '.mi')

    mi.save(save_dir)


