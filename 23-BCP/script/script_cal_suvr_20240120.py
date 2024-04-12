import numpy as np
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import openpyxl
import time

from xomics import MedicalImage



def cal_suv_mean(image, label, cal_reference: bool, plt_hist: bool= False):
  '''
  output: mean, weight
  '''
  assert image.shape == label.shape
  image = image * label
  image = np.sort(image[image != 0])

  if not cal_reference:
    return np.mean(image), len(image)
  else:
    percentile_low, percentile_high = 25, 75

    low_value = np.percentile(image, percentile_low)
    high_value = np.percentile(image, percentile_high)

    selected_data = image[(image >= low_value) & (image <= high_value)]

    # Show Hist
    if plt_hist:
      plt.hist(image.flatten(), bins=50, alpha=0.7, label='Full Range')
      hist, edges = np.histogram(image.flatten(), bins=50)
      start_index = np.argmax(edges >= low_value)
      end_index = np.argmax(edges >= high_value)
      plt.bar(edges[start_index:end_index], hist[start_index:end_index], color='orange', width=100)

      plt.xlabel('Values')
      plt.ylabel('Frequency')
      plt.legend()
      plt.show()

    return np.mean(selected_data), len(selected_data)


def cal_suvr_combined_braak(braak_mean_suv, braak_weight, reference_suv):
  whole_suv = np.sum([s * w for s, w in zip(braak_mean_suv, braak_weight)])
  whole_weight = np.sum(braak_weight)

  return whole_suv / whole_weight / reference_suv


def create_new_sitk_image(array: np.ndarray, template_image: sitk.Image):
  new_image = sitk.GetImageFromArray(array)
  new_image.SetDirection(template_image.GetDirection())
  new_image.SetSpacing(template_image.GetSpacing())
  new_image.SetOrigin(template_image.GetOrigin())
  return new_image


def get_region_reference_num_from_DK_atlas():
  # whole cerebellum: [7, 46, 8, 47]
  # cerebellum without graymatter: [8, 47]
  atlas_dict = {}

  reference_num = [8, 47]

  region_num_braak1_left = [1006]
  region_num_braak1_right = [2006]

  region_num_braak2_left = [17]
  region_num_braak2_right = [53]

  region_num_braak3_left = [1016, 1007, 1013, 18]
  region_num_braak3_right = [2016, 2007, 2013, 54]

  region_num_braak4_left = [1015, 1002, 1026, 1023, 1010, 1035, 1009, 1033]
  region_num_braak4_right = [2015, 2002, 2026, 2023, 2010, 2035, 2009, 2033]

  region_num_braak5_left = [1012, 1014, 1032, 1003, 1027, 1018, 1019, 1020, 1011, 1031, 1008, 1030, 1029, 1025, 1001, 1034, 1028]
  region_num_braak5_right = [2012, 2014, 2032, 2003, 2027, 2018, 2019, 2020, 2011, 2031, 2008, 2030, 2029, 2025, 2001, 2034, 2028]

  region_num_braak6_left = [1021, 1022, 1005, 1024, 1017]
  region_num_braak6_right = [2021, 2022, 2005, 2024, 2017]

  atlas_dict['reference'] = reference_num

  atlas_dict['b1_left'] = region_num_braak1_left
  atlas_dict['b2_left'] = region_num_braak2_left
  atlas_dict['b3_left'] = region_num_braak3_left
  atlas_dict['b4_left'] = region_num_braak4_left
  atlas_dict['b5_left'] = region_num_braak5_left
  atlas_dict['b6_left'] = region_num_braak6_left

  atlas_dict['b1_right'] = region_num_braak1_right
  atlas_dict['b2_right'] = region_num_braak2_right
  atlas_dict['b3_right'] = region_num_braak3_right
  atlas_dict['b4_right'] = region_num_braak4_right
  atlas_dict['b5_right'] = region_num_braak5_right
  atlas_dict['b6_right'] = region_num_braak6_right

  return atlas_dict



if __name__ == '__main__':
  work_dir = r'E:\xai-omics\data\30-Brain-SQA\2024-01-20-77\outputs_20240120_suv'
  pids = [f for f in os.listdir(work_dir)
          if os.path.isdir(os.path.join(work_dir, f)) and f != 'mi']
  segment_model = 'fastsurfer'
  save_dir = r'E:\xai-omics\data\30-Brain-SQA\2024-01-20-77\results'
  excel_name = '2024-01-20-tau(suv correction)_test.xlsx'
  save_path = os.path.join(save_dir, excel_name)

  print("', '".join(pids))

  time_total_start = time.time()
  excel_data = {}
  for p_index, p in enumerate(pids):
    patient_path = os.path.join(work_dir, p)
    pd_type = ['ab', 'tau']
    p_time_start = time.time()

    for t in pd_type:
      if t == 'ab': continue
      mask_path = os.path.join(patient_path, f'dk_mask_{segment_model}.nii')
      mr_path = os.path.join(patient_path, f'{segment_model}_mr.nii')
      pet_path = os.path.join(patient_path, f'cor_{t}_pet.nii')
      if not os.path.exists(pet_path): continue

      norm_path = os.path.join(patient_path, 'norm_and_mask')
      if not os.path.exists(norm_path): os.mkdir(norm_path)

      norm_pet_path = os.path.join(norm_path, f'norm_{t}_pet.nii')
      norm_masked_pet_path = os.path.join(norm_path, f'norm_{t}_masked_pet.nii')
      all_mask_path = os.path.join(norm_path, f'{segment_model}_{p}_{t}_all_mask.nii')

      # Reading NITFI files
      mask = sitk.ReadImage(mask_path)
      mr = sitk.ReadImage(mr_path)
      pet = sitk.ReadImage(pet_path)

      # Get Array from NITFI
      mask_array = sitk.GetArrayFromImage(mask)
      mr_array = sitk.GetArrayFromImage(mr)
      pet_array = sitk.GetArrayFromImage(pet)

      # Calculate SUVr in SynthSeg-Based Method
      atlas_dict = get_region_reference_num_from_DK_atlas()

      region_mask, reference_mask = mask_array, mask_array
      # Reference
      reference_mask = np.isin(reference_mask, atlas_dict['reference']).astype(int)

      # Left and Right
      num_half = [atlas_dict[k] for k in atlas_dict.keys() if k != 'reference']
      region_mask_braak_half = [np.isin(region_mask, num).astype(int)
                                for num in num_half]

      region_suv_braak_half, region_weight_braak_half = zip(*[cal_suv_mean(
        pet_array, rmb, False) for rmb in region_mask_braak_half])

      reference_suv, _ = cal_suv_mean(pet_array, reference_mask, True)
      suvr_braak_half = [rsb / reference_suv for rsb in region_suv_braak_half]

      # Single
      suvr_braak_single = [
        cal_suvr_combined_braak(
          [region_suv_braak_half[b], region_suv_braak_half[b + 6]],
          [region_weight_braak_half[b], region_weight_braak_half[b + 6]],
          reference_suv) for b in range(6)]

      # Both
      xx = [x for x in range(0, 6, 2)]
      suvr_braak_both = [
        cal_suvr_combined_braak(
          region_suv_braak_half[b:b + 2] + region_suv_braak_half[b + 6: b + 8],
          region_weight_braak_half[b:b + 2] + region_weight_braak_half[b + 6:b + 8],
          reference_suv) for b in range(0, 6, 2)]

      # Norm pet nii
      norm_pet_array = pet_array / reference_suv
      # norm_pet_array[norm_pet_array < 0] = 0
      # new_pet_array[new_pet_array > 3] = 3
      norm_pet = create_new_sitk_image(norm_pet_array, pet)
      sitk.WriteImage(norm_pet, norm_pet_path)

      # Norm masked pet
      norm_masked_pet_array = norm_pet_array * (np.sum(region_mask_braak_half + [reference_mask]))
      # new_pet_mask[0, 0, 0] = 3
      norm_masked_pet = create_new_sitk_image(norm_masked_pet_array, pet)
      sitk.WriteImage(norm_masked_pet, norm_masked_pet_path)

      # All mask
      for i in range(len(region_mask_braak_half)):
        region_mask_braak_half[i][region_mask_braak_half[i] == 1] = i + 1

      reference_mask[reference_mask == 1] = 20
      all_mask_array = np.sum(region_mask_braak_half + [reference_mask], axis=0)
      all = create_new_sitk_image(all_mask_array, mask)

      sitk.WriteImage(all, all_mask_path)

      excel_data[p] = ([p] +
                       list(np.round(suvr_braak_half, 4)) +
                       list(np.round(suvr_braak_single, 4)) +
                       list(np.round(suvr_braak_both, 4)) +
                       list(np.round(region_suv_braak_half, 4)) +
                       list(np.round(region_weight_braak_half)) +
                       list(np.round([reference_suv], 4)))
      # print('\t\t\t\t'.join(['braak1', 'braak2', 'braak3', 'braak4', 'braak5', 'braak6', 'reference']))
      # print('\t\t\t\t'.join(map(lambda x: f'{int(x)}', region_suv_braak + [reference_suv])))
      # print('\t\t\t\t'.join(map(lambda x: f'{x: .2f}', suvr_braak)))

    p_time_end = time.time()
    p_time = p_time_end - p_time_start
    time_total = p_time_end - time_total_start
    print(f"[{p_index + 1}/{len(pids)}] PIDS:{p}\t\t\t\t\t\t"
          f"Consuming Time: {p_time: .2f} s\t\t\t"
          f"total: {time_total: .2f} s")

  # Save as excel
  workbook = openpyxl.Workbook()
  sheet = workbook.active
  sheet.append([
    'PIDS',
    'SUVR-B1-L', 'SUVR-B2-L', 'SUVR-B3-L', 'SUVR-B4-L', 'SUVR-B5-L', 'SUVR-B6-L',
    'SUVR-B1-R', 'SUVR-B2-R', 'SUVR-B3-R', 'SUVR-B4-R', 'SUVR-B5-R', 'SUVR-B6-R',
    'SUVR-B1', 'SUVR-B2', 'SUVR-B3', 'SUVR-B4', 'SUVR-B5', 'SUVR-B6',
    'SUVR-B12', 'SUVR-B34', 'SUVR-B56',
    'SUV-B1-L', 'SUV-B2-L', 'SUV-B3-L', 'SUV-B4-L', 'SUV-B5-L', 'SUV-B6-L',
    'SUV-B1-R', 'SUV-B2-R', 'SUV-B3-R', 'SUV-B4-R', 'SUV-B5-R', 'SUV-B6-R',
    'Weight-B1-L', 'Weight-B2-L', 'Weight-B3-L',
    'Weight-B4-L', 'Weight-B5-L', 'Weight-B6-L',
    'Weight-B1-R', 'Weight-B2-R', 'Weight-B3-R',
    'Weight-B4-R', 'Weight-B5-R', 'Weight-B6-R',
    'SUV-Reference'])

  for key in excel_data:
    sheet.append(excel_data[key])

  workbook.save(save_path)
  print(f'Excel file saved to {save_path}')
