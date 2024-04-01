import numpy as np
import SimpleITK as sitk
import os

from tqdm import tqdm
from scipy.ndimage import label, generate_binary_structure



def create_new_sitk_image(array: np.ndarray, template_image: sitk.Image):
	new_image = sitk.GetImageFromArray(array)
	new_image.SetDirection(template_image.GetDirection())
	new_image.SetSpacing(template_image.GetSpacing())
	new_image.SetOrigin(template_image.GetOrigin())
	return new_image



if __name__ == '__main__':
	work_dir = r'E:\xai-omics\data\31-Prostate'
	image_dir = os.path.join(work_dir, 'mask_and_image')
	mask_dir = os.path.join(work_dir, 'mask_and_image')
	save_dir = image_dir
	save_name = 'lesion.nii.gz'

	pids = os.listdir(image_dir)

	# Parameter Setting
	percentile = 99.9
	pad_width = 10

	# Start
	for pid in tqdm(pids):
		# if '0245' not in pid: continue
		# Path Setting
		pet_path = os.path.join(image_dir, pid, 'pet.nii.gz')
		mask_path = os.path.join(mask_dir, pid, 'new_prostate_demo3.nii.gz')
		save_path = os.path.join(save_dir, pid, save_name)

		# Reading Image from nii files
		pet_image = sitk.ReadImage(pet_path)
		mask_image = sitk.ReadImage(mask_path)

		# Get array from sitk.Image
		pet_array = sitk.GetArrayFromImage(pet_image)
		mask_array = sitk.GetArrayFromImage(mask_image)
		data_shape = mask_array.shape

		# Crop size
		indice = np.where(mask_array == 1)[0]
		top = min(np.max(indice) + pad_width, data_shape[0])
		bottom = max(np.min(indice) - pad_width, 0)

		crop_pet_array = pet_array[bottom:top]
		crop_mask_array = mask_array[bottom:top]

		assert np.min(crop_pet_array)	>= 0
		crop_pet_array[crop_mask_array == 0] = -1

		threshold = np.percentile(crop_pet_array, q=percentile)
		crop_pet_array = (crop_pet_array >= threshold).astype(np.uint8)

		new_mask_array = crop_pet_array * crop_mask_array

		new_mask_array = np.pad(
			new_mask_array, ((bottom, data_shape[0] - top), (0, 0), (0, 0)),
			'constant', constant_values=0)

		new_mask_image = create_new_sitk_image(new_mask_array, mask_image)
		sitk.WriteImage(new_mask_image, save_path)

	print()