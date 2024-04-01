import numpy as np
import SimpleITK as sitk
import os
import cv2

from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure, gaussian_filter



def create_new_sitk_image(array: np.ndarray, template_image: sitk.Image):
	new_image = sitk.GetImageFromArray(array)
	new_image.SetDirection(template_image.GetDirection())
	new_image.SetSpacing(template_image.GetSpacing())
	new_image.SetOrigin(template_image.GetOrigin())
	return new_image


def apply_3d_gaussian_filter(image, sigma=1):
	filtered_image = gaussian_filter(image, sigma=sigma)
	return filtered_image



if __name__ == '__main__':
	work_dir = r'E:\xai-omics\data\31-Prostate'
	image_dir = os.path.join(work_dir, 'mask_and_image')
	mask_dir = os.path.join(work_dir, 'mask_all')
	save_dir = image_dir
	save_dir2 = os.path.join(work_dir, 'mask_demo3')
	save_name = 'new_prostate_demo3.nii.gz'

	pids = os.listdir(image_dir)[-41:]

	# Parameter Setting
	percentile = 99.9
	pad_width = 10
	size_ratio = 0.3
	urinary_bladder_index = 21
	prostate_index = 22
	threshold_ct = 25

	# Start
	for pid in tqdm(pids):
		# if '0041' not in pid: continue
		# Path Setting
		ct_path = os.path.join(image_dir, pid, 'ct.nii.gz')
		pet_path = os.path.join(image_dir, pid, 'pet.nii.gz')
		mask_path = os.path.join(mask_dir, pid, 'mask.nii.gz')
		save_path = os.path.join(save_dir, pid, save_name)

		# Reading Image from nii files
		ct_image = sitk.ReadImage(ct_path)
		pet_image = sitk.ReadImage(pet_path)
		mask_image = sitk.ReadImage(mask_path)

		# Get array from sitk.Image
		ct_array = sitk.GetArrayFromImage(ct_image)
		pet_array = sitk.GetArrayFromImage(pet_image)
		mask_array = sitk.GetArrayFromImage(mask_image)
		data_shape = mask_array.shape

		# Get urinary bladder mask and prostate mask
		urinary_bladder_array = (mask_array == urinary_bladder_index).astype(np.uint8)
		prostate_array = (mask_array == prostate_index).astype(np.uint8)

		# Crop size
		indice = np.where(prostate_array == 1)[0]
		top = min(np.max(indice) + pad_width, data_shape[0])
		bottom = max(np.min(indice) - pad_width, 0)

		crop_pet_array = pet_array[bottom:top]
		crop_prostate_array = prostate_array[bottom:top]
		crop_ub_array = urinary_bladder_array[bottom:top]

		threshold = np.percentile(crop_pet_array, q=percentile)
		crop_pet_array = (crop_pet_array >= threshold).astype(np.uint8)

		# Calculate connection region
		structure = generate_binary_structure(3, 3)
		labeled_array, num_features = label(crop_pet_array, structure)

		# region: xxxx
		ct_ub_array = apply_3d_gaussian_filter(ct_array, 1)
		ct_ub_array[urinary_bladder_array == 0] = 3000
		ct_ub_array[ct_ub_array < threshold_ct] = 0
		ct_ub_array[ct_ub_array != 0] = 1

		test_array = (~ct_ub_array.astype(bool)).astype(np.uint8)

		# Test
		test_labeled_array, test_num_features = label(test_array, structure)
		test_connection_num, test_connection_vol = np.unique(test_labeled_array, return_counts=True)
		test_volume_dict = dict(zip(test_connection_num, test_connection_vol))
		test_volume_dict.pop(0, None)

		max_key = max(test_volume_dict, key=test_volume_dict.get)
		test_labeled_array[test_labeled_array != max_key] = 0
		test_labeled_array[test_labeled_array == max_key] = 1
		test_array = test_labeled_array.astype(np.uint8)

		test_image = create_new_sitk_image(test_array, mask_image)
		sitk.WriteImage(test_image, os.path.join(save_dir, pid, 'ub_true_mask.nii.gz'))

		ct_ub_array = (~test_array.astype(bool)).astype(np.uint8)

		crop_ct_ub_array = ct_ub_array[bottom:top]

		labeled_array = labeled_array * crop_ct_ub_array
		# endregion: xxxx

		connection_num, connection_vol = np.unique(labeled_array, return_counts=True)
		volume_dict = dict(zip(connection_num, connection_vol))
		volume_dict.pop(0, None)

		connection_pro_num, connection_pro_vol = np.unique(
			crop_prostate_array * labeled_array, return_counts=True)
		volume_pro_dict = dict(zip(connection_pro_num, connection_pro_vol))
		volume_pro_dict.pop(0, None)

		connection_ub_num, connection_ub_vol = np.unique(
			crop_ub_array * labeled_array, return_counts=True)
		volume_ub_dict = dict(zip(connection_ub_num, connection_ub_vol))
		volume_ub_dict.pop(0, None)

		num_label = []
		for c, v in zip(connection_num, connection_vol):
			if c == 0 or c not in volume_pro_dict.keys(): continue
			if c not in volume_ub_dict.keys():
				num_label.append(c)
				continue

			v_all = volume_dict[c]
			v_pro = volume_pro_dict[c]
			v_ub = volume_ub_dict[c]

			if v_all < 200 or v_pro < 200: continue
			# if v_pro < v_ub * 0.7 or v_pro < v_all * size_ratio: continue
			num_label.append(c)

		if len(num_label) == 0: print(pid)

		bool_array = np.isin(labeled_array, num_label)
		labeled_array = np.where(bool_array, 1, 0)

		labeled_array = np.pad(
			labeled_array, ((bottom, data_shape[0] - top), (0, 0), (0, 0)),
			'constant', constant_values=0)

		new_mask_array = labeled_array + prostate_array
		new_mask_array[new_mask_array != 0] = 1

		new_mask_image = create_new_sitk_image(new_mask_array, mask_image)
		sitk.WriteImage(new_mask_image, save_path)

		save_path2 = os.path.join(save_dir2, pid)
		if not os.path.exists(save_path2): os.makedirs(save_path2)
		sitk.WriteImage(new_mask_image, os.path.join(save_path2, 'prostate_demo3.nii.gz'))

	print()