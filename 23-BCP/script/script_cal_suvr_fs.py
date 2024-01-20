import numpy as np
import SimpleITK as sitk
import os

from xomics import MedicalImage
import matplotlib.pyplot as plt



def cal_sur_mean(image, label, cal_reference: bool):
  assert image.shape == label.shape
  image = image * label
  image = np.sort(image[image != 0])

  if not cal_reference:
    return np.mean(image)
  else:
    percentile_low, percentile_high = 25, 75

    low_value = np.percentile(image, percentile_low)
    high_value = np.percentile(image, percentile_high)

    selected_data = image[(image >= low_value) & (image <= high_value)]

    # Show Hist
    plt.hist(image.flatten(), bins=50, alpha=0.7, label='Full Range')
    hist, edges = np.histogram(image.flatten(), bins=50)
    start_index = np.argmax(edges >= low_value)
    end_index = np.argmax(edges >= high_value)
    plt.bar(edges[start_index:end_index], hist[start_index:end_index], color='orange', width=100)

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    return np.mean(selected_data)


def cal_sur_mean_backup(image, label):
  assert image.shape == label.shape
  return np.sum(image * label) / np.sum(label)



if __name__ == '__main__':
  input_dir = r'E:\BrainSeg'
  segment_model = ['freesurfer', 'fastsurfer'][1]
  input_dir = os.path.join(input_dir, segment_model)
  pids = [i for i in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, i)) and i != 'mi']

  for p in pids:
    patient_path = os.path.join(input_dir, p)
    av_type = ['45', '1451']

    for t in av_type:
      mask_path = os.path.join(patient_path, f'dk_mask_{segment_model}.nii')

      mri_path = os.path.join(patient_path, f'{segment_model}_mr.nii')
      pet_path = os.path.join(patient_path, f'cor_{t}_pet.nii')

      norm_path = os.path.join(patient_path, 'norm_and_mask')
      if not os.path.exists(norm_path): os.mkdir(norm_path)

      norm_pet_path = os.path.join(norm_path, f'norm_{t}_pet.nii')
      norm_masked_pet_path = os.path.join(norm_path, f'norm_{t}_masked_pet.nii')
      both_mask_path = os.path.join(norm_path, f'{segment_model}_{p}_{t}_both_mask.nii')

      # Reading NITFI files
      mask = sitk.ReadImage(mask_path)
      mri = sitk.ReadImage(mri_path)
      pet = sitk.ReadImage(pet_path)

      # Get Array from NITFI
      mask_array = sitk.GetArrayFromImage(mask)
      mri_array = sitk.GetArrayFromImage(mri)
      pet_array = sitk.GetArrayFromImage(pet)

      # Calculate SUVr in SynthSeg-Based Method
      if t == '45':
        region_num = [1005, 1007, 1008, 1009, 1011, 1013, 1015, 1017, 1021, 1022, 1025, 1029, 1030, 1031, 1034,
                      2005, 2007, 2008, 2009, 2011, 2013, 2015, 2017, 2021, 2022, 2025, 2029, 2030, 2031, 2034]
        region_num = [1015, 2015, 1013, 2013]
        region_num = [1002, 1003, 1008, 1009, 1010, 1011, 1012, 1014, 1015, 1023, 1026, 1027, 1028, 1029, 1030, 1034,
                      2002, 2003, 2008, 2009, 2010, 2011, 2012, 2014, 2015, 2023, 2026, 2027, 2028, 2029, 2030, 2034]
        reference_num = [7, 46, 8, 47]
      else:
        region_num = [1006, 2006, 1016, 2016, 18, 54, 1009, 2009, 1015, 2015]
        reference_num = [8, 47]

      region_mask, reference_mask = mask_array, mask_array
      region_mask = np.isin(region_mask, region_num).astype(int)
      reference_mask = np.isin(reference_mask, reference_num).astype(int)

      # Calculate SUVr
      region_suv = cal_sur_mean(pet_array, region_mask, False)
      reference_suv = cal_sur_mean(pet_array, reference_mask, True)
      print(f'region_suv: {region_suv} ----- reference_suv: {reference_suv}')

      suvr = region_suv / reference_suv
      print(f'{p} --- {t} --- {suvr: .2f}')

      # Create new pet nii and pet_mask
      new_pet_array = pet_array / reference_suv
      new_pet_array[new_pet_array < 0] = 0
      new_pet_array[new_pet_array > 3] = 3

      new_pet = sitk.GetImageFromArray(new_pet_array)
      new_pet.SetSpacing(pet.GetSpacing())
      new_pet.SetOrigin(pet.GetOrigin())
      new_pet.SetDirection(pet.GetDirection())

      sitk.WriteImage(new_pet, norm_pet_path)

      new_pet_mask = new_pet_array * (region_mask + reference_mask)
      new_pet_mask[0, 0, 0] = 3

      new_mask = sitk.GetImageFromArray(new_pet_mask)
      new_mask.SetSpacing(pet.GetSpacing())
      new_mask.SetOrigin(pet.GetOrigin())
      new_mask.SetDirection(pet.GetDirection())

      sitk.WriteImage(new_mask, norm_masked_pet_path)

      # Create mask nii
      reference_mask[reference_mask == 1] = 2
      both_mask_array = region_mask + reference_mask

      both = sitk.GetImageFromArray(both_mask_array)
      both.SetSpacing(mask.GetSpacing())
      both.SetDirection(mask.GetDirection())
      both.SetOrigin(mask.GetOrigin())

      sitk.WriteImage(both, both_mask_path)

    print()

