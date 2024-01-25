import numpy as np
import SimpleITK as sitk
import os

from xomics import MedicalImage



def cal_sur_mean(image, label):
  assert image.shape == label.shape
  return np.sum(image * label) / np.sum(label)



if __name__ == '__main__':
  input_dir = r'E:\BrainSeg\data'
  pid = [i for i in os.listdir(input_dir) if i[-4:] != '.nii']


  for p in pid:
    patient_path = os.path.join(input_dir, p)
    av_type = ['45', '1451']

    for t in av_type:
      av_path = os.path.join(patient_path, t)
      synthseg_pet_path = os.path.join(av_path, 'rpet.nii')
      synthseg_mask_path = os.path.join(av_path, 'rmask_s1.nii')
      spm_pet_path = os.path.join(av_path, 'rwpet.nii')
      spm_region_mask_path = os.path.join(input_dir, f'av{t}_spm_region.nii')
      spm_reference_mask_path = os.path.join(input_dir, f'av{t}_spm_reference.nii')

      # Reading NITFI files
      synthseg_pet = sitk.ReadImage(synthseg_pet_path)
      synthseg_mask = sitk.ReadImage(synthseg_mask_path)
      spm_pet = sitk.ReadImage(spm_pet_path)
      spm_region_mask = sitk.ReadImage(spm_region_mask_path)
      spm_reference_mask = sitk.ReadImage(spm_reference_mask_path)

      # Get Array from NITFI
      synthseg_pet = sitk.GetArrayFromImage(synthseg_pet)
      synthseg_mask = sitk.GetArrayFromImage(synthseg_mask)
      spm_pet = sitk.GetArrayFromImage(spm_pet)
      spm_region_mask = sitk.GetArrayFromImage(spm_region_mask)
      spm_reference_mask = sitk.GetArrayFromImage(spm_reference_mask)

      # Calculate SUVr in SPM-Based Method
      suvr = cal_sur_mean(spm_pet, spm_region_mask) / cal_sur_mean(spm_pet, spm_reference_mask)
      print(f'{p} -- {t} -- SPM -- {suvr: .2f}')

      # Calculate SUVr in SynthSeg-Based Method
      # if t == '45':
      #   region_num = [3, 42]
      #   reference_num = [7, 8, 46, 47]
      # else:
      #   region_num = [3, 42, 18, 54]
      #   reference_num = [8, 47]
      #
      # region_mask, reference_mask = synthseg_mask, synthseg_mask
      # region_mask = np.isin(region_mask, region_num).astype(int)
      # reference_mask = np.isin(reference_mask, reference_num).astype(int)
      #
      # suvr = cal_sur_mean(synthseg_pet, region_mask) / cal_sur_mean(synthseg_pet, reference_mask)
      # print(f'{p} -- {t} -- SynthSeg -- {suvr: .4f}')
      #
      # print()