import os
import SimpleITK as sitk
import numpy as np
import time

from xomics import MedicalImage
from tqdm import tqdm



def save_as_mi(ct_image, pet_image, mask_image, mi_save_path):
	array_ct = sitk.GetArrayFromImage(ct_image)
	array_pet = sitk.GetArrayFromImage(pet_image)
	array_mask = sitk.GetArrayFromImage(mask_image)

	assert array_ct.shape == array_pet.shape == array_mask.shape
	pid = os.path.basename(mi_save_path).split('.mi')[0]
	mi = MedicalImage(
		images={'ct': array_ct, 'pet': array_pet},
		labels={'region': array_mask}, key=pid)

	mi.put_into_pocket(mi.Keys.SPACING, ct_image.GetSpacing(), local=True)
	mi.put_into_pocket(mi.Keys.DIRECTION, ct_image.GetDirection(), local=True)
	mi.put_into_pocket(mi.Keys.PATIENTNAME, pid, local=True)

	mi.save(mi_save_path)



if __name__ == '__main__':
	'''
	Convert ct.nii.gz, pet.nii.gz and prostate.nii.gz from TotalSegmentator to
	MI files
	'''
	# Input
	work_dir = r'E:\xai-omics\data\31-Prostate'
	image_dir = os.path.join(work_dir, 'mask_and_image')
	patient_ids = os.listdir(image_dir)
	mask_dir = os.path.join(work_dir, 'mask_and_image')

	# Output
	save_dir = os.path.join(work_dir, 'mi')

	# Parameter
	time_total_start = time.time()

	for p_index, p in enumerate(patient_ids):
		p_time_start = time.time()
		ct_path = os.path.join(image_dir, p, 'ct.nii.gz')
		pet_path = os.path.join(image_dir, p, 'pet_aorta.nii.gz')
		mask_path = os.path.join(mask_dir, p, 'lesion.nii.gz')

		ct_image = sitk.ReadImage(ct_path)
		pet_image = sitk.ReadImage(pet_path)
		mask_image = sitk.ReadImage(mask_path)

		save_as_mi(
			ct_image, pet_image, mask_image, os.path.join(save_dir, p + '.mi'))

		p_time_end = time.time()
		p_time = p_time_end - p_time_start
		time_total = p_time_end - time_total_start
		print(
			f"[{p_index + 1}/{len(patient_ids)}] PIDS:{p}\t\t\t\t\t\t"
			f"Consuming Time: {p_time: .2f} s\t\t\t"
			f"total: {time_total: .2f} s")
