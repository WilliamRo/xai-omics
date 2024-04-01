import os
import time
import SimpleITK as sitk
import numpy as np

from xomics import MedicalImage



if __name__ == '__main__':
	'''
	Convert raw mi file to nii file for feature extraction
	1. Load mi file
	2. CT image: window, normalize to [-1, 1]
	3. PET image: z-score normalization
	4. CT, PET and Mask are cropped to [64, 64, 64]
	5. Save as nii file
	'''
	# Input
	work_dir = r'E:\xai-omics\data\31-Prostate'
	mi_dir = os.path.join(work_dir, 'mi')
	mi_files = os.listdir(mi_dir)[50:]

	# Output
	save_dir = os.path.join(work_dir, 'nii_128')

	# Parameter
	time_total_start = time.time()
	# crop_size = [256, 256, 256]
	crop_size = [128, 128, 128]
	# crop_size = [64, 64, 64]
	window = [-300, 400]

	for p_index, f in enumerate(mi_files):
		p_time_start = time.time()

		mi = MedicalImage.load(os.path.join(mi_dir, f))

		mi.window('ct', window[0], window[1])

		mi.normalization(['ct'], 'min_max')
		mi.images['ct'] = mi.images['ct'] * 2 - 1
		mi.images['ct'] = mi.images['ct'].astype(np.float32)
		mi.normalization(['pet'], 'z_score')

		_ = mi.crop(crop_size, False, basis=['region'])

		ct_array, pet_array, mask_array = mi.images['ct'], mi.images['pet'], mi.labels['region']
		# Test
		space = mi._local_pocket[mi.Keys.SPACING]
		direction = mi._local_pocket[mi.Keys.DIRECTION]
		pid = mi._local_pocket[mi.Keys.PATIENTNAME]

		# space = mi.get_from_pocket(mi.Keys.SPACING, local=True)
		# direction = mi.get_from_pocket(mi.Keys.DIRECTION, local=True)
		# pid = mi.get_from_pocket(mi.Keys.PATIENTNAME, local=True)

		ct_image = sitk.GetImageFromArray(ct_array)
		pet_image = sitk.GetImageFromArray(pet_array)
		mask_image = sitk.GetImageFromArray(mask_array)

		ct_image.SetSpacing(space)
		pet_image.SetSpacing(space)
		mask_image.SetSpacing(space)

		ct_image.SetDirection(direction)
		pet_image.SetDirection(direction)
		mask_image.SetDirection(direction)

		save_path = os.path.join(save_dir, pid)
		if not os.path.exists(save_path): os.mkdir(save_path)

		sitk.WriteImage(ct_image, os.path.join(save_path, 'ct.nii.gz'))
		sitk.WriteImage(pet_image, os.path.join(save_path, 'pet.nii.gz'))
		sitk.WriteImage(mask_image, os.path.join(save_path, 'mask.nii.gz'))

		p_time_end = time.time()
		p_time = p_time_end - p_time_start
		time_total = p_time_end - time_total_start
		print(
			f"[{p_index + 1}/{len(mi_files)}] PIDS:{pid}\t\t\t\t\t\t"
			f"Consuming Time: {p_time: .2f} s\t\t\t"
			f"total: {time_total: .2f} s")