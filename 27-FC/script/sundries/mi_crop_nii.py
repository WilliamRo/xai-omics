import SimpleITK as sitk
import numpy as np
import os

from xomics import MedicalImage
from tqdm import tqdm



if __name__ == "__main__":
	'''
	输入是dcm2mi后的mi文件，这里将mi文件中的ct和pet都crop后，再做normalization，并保存成nii格式。
	'''
	work_dir = r'E:\xai-omics\data\02-PET-CT-Y1\mi\ecs_region'
	save_dir = r'E:\xai-omics\data\02-PET-CT-Y1\nii\ecs_region'

	mi_file_list = [os.path.join(work_dir, f) for f in os.listdir(work_dir)]

	for f in tqdm(mi_file_list, desc='Loading mi files'):
		mi: MedicalImage = MedicalImage.load(f)
		mi.window('ct', -300, 400)

		_, _ = mi.crop([128, 256, 256], random_crop=False, basis=['lesion_gt'])

		mi.normalization(['ct'], 'min_max')
		mi.images['ct'] = mi.images['ct'] * 2 - 1
		mi.images['ct'] = mi.images['ct'].astype(np.float32)

		mi.images['pet'] = (mi.images['pet'] - np.mean(mi.images['pet'])) / np.std(mi.images['pet'])

		ct = sitk.GetImageFromArray(mi.images['ct'])
		ct.SetSpacing(mi.get_from_pocket('space'))
		ct.SetDirection(mi.get_from_pocket('direction'))

		pet = sitk.GetImageFromArray(mi.images['pet'])
		pet.SetSpacing(mi.get_from_pocket('space'))
		pet.SetDirection(mi.get_from_pocket('direction'))

		save_path = os.path.join(save_dir, mi.get_from_pocket('patient_name'))
		if not os.path.exists(save_path): os.mkdir(save_path)

		sitk.WriteImage(ct, os.path.join(save_path, 'ct.nii'))
		sitk.WriteImage(pet, os.path.join(save_path, 'pet.nii'))
