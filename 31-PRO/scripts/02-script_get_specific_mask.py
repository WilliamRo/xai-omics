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
	'''
	Get the specific mask from 117 masks
	'''
	work_dir = r'E:\xai-omics\data\31-Prostate'
	image_dir = os.path.join(work_dir, 'mask_and_image')
	mask_dir = os.path.join(work_dir, 'mask_all')
	save_dir = image_dir

	pids = os.listdir(image_dir)[-41:]

	# Parameter Setting
	specific_mask_index = [52]
	save_name = 'mask_aorta.nii.gz'

	# Start
	for pid in tqdm(pids):
		# Path Setting
		mask_path = os.path.join(mask_dir, pid, 'mask.nii.gz')
		save_path = os.path.join(save_dir, pid, save_name)

		# Reading Image from nii files
		mask_image = sitk.ReadImage(mask_path)

		# Get array from sitk.Image
		mask_array = sitk.GetArrayFromImage(mask_image)

		new_mask_array = np.zeros_like(mask_array)
		new_mask_array[np.isin(mask_array, specific_mask_index)] = 1

		new_mask_image = create_new_sitk_image(new_mask_array, mask_image)
		sitk.WriteImage(new_mask_image, save_path)