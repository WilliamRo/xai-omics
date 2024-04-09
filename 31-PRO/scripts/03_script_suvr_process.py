import os
import SimpleITK as sitk
import numpy as np
import time

from xomics import MedicalImage
from tqdm import tqdm



def create_new_sitk_image(array: np.ndarray, template_image: sitk.Image):
	new_image = sitk.GetImageFromArray(array)
	new_image.SetDirection(template_image.GetDirection())
	new_image.SetSpacing(template_image.GetSpacing())
	new_image.SetOrigin(template_image.GetOrigin())
	return new_image



if __name__ == '__main__':
	'''
	Preprocess in PET image
	1. Choose a body part e.g. liver or aorta
	2. Calculate the SUV_mean in this body part (aorta)
	3. PET Image / SUV_mean 
	'''
	# Input and Output
	work_dir = r'E:\xai-omics\data\31-Prostate'
	image_dir = os.path.join(work_dir, 'mask_and_image')
	mask_dir = image_dir
	save_dir = image_dir

	# Parameter Setting
	mask_name = 'mask_aorta.nii.gz'
	save_name = 'pet' + mask_name.split('mask')[-1]
	pids = os.listdir(image_dir)

	# Start
	for pid in tqdm(pids):
		# if '0245' not in pid: continue
		# Path Setting
		pet_path = os.path.join(image_dir, pid, 'pet.nii.gz')
		mask_path = os.path.join(mask_dir, pid, mask_name)
		save_path = os.path.join(save_dir, pid, save_name)

		# Reading Image from nii files
		pet_image = sitk.ReadImage(pet_path)
		mask_image = sitk.ReadImage(mask_path)

		# Get array from sitk.Image
		pet_array = sitk.GetArrayFromImage(pet_image)
		mask_array = sitk.GetArrayFromImage(mask_image)

		suv_mean = np.mean(pet_array[mask_array == 1])
		suv_std = np.std(pet_array[mask_array == 1])
		print(f'{pid}\t{suv_mean: .2f}\t{suv_std: .2f}')
		new_pet_array = pet_array / suv_mean

		new_pet_image = create_new_sitk_image(new_pet_array, mask_image)
		sitk.WriteImage(new_pet_image, save_path)
