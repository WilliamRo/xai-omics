import SimpleITK as sitk
import numpy as np
import os
import time

from tqdm import tqdm



def dcm2nii(dcm_dir) -> sitk.Image:
	reader = sitk.ImageSeriesReader()
	dcm_names = reader.GetGDCMSeriesFileNames(dcm_dir)
	reader.SetFileNames(dcm_names)
	image = reader.Execute()

	return image


def reorient(image:sitk.Image) -> sitk.Image:
	direction = image.GetDirection()
	direction = tuple(round(d) for d in direction)
	image.SetDirection(direction)

	return image


def reorigin(image:sitk.Image) -> sitk.Image:
	array = sitk.GetArrayFromImage(image)

	head = image.TransformIndexToPhysicalPoint([0, 0, 0])
	tail = image.TransformIndexToPhysicalPoint(array.shape[::-1])

	new_origin = tuple((h - t) / 2 for h, t in zip(head, tail))
	image.SetOrigin(new_origin)

	return image


def resample(image: sitk.Image, outspacing: list=[1, 1, 1]) -> sitk.Image:
	inputsize = image.GetSize()
	inputspacing = image.GetSpacing()

	transform = sitk.Transform()
	transform.SetIdentity()

	outsize = [
		int(inputsize[i] * inputspacing[i] / outspacing[i] + 0.5)
		for i in range(3)]

	resampler = sitk.ResampleImageFilter()
	resampler.SetTransform(transform)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetOutputOrigin(image.GetOrigin())
	resampler.SetOutputSpacing(outspacing)
	resampler.SetOutputDirection(image.GetDirection())

	# new_out_size = [int(2 ** np.ceil(np.log2(size))) for size in outsize]
	# resampler.SetSize(new_out_size)
	resampler.SetSize(outsize)
	new_image = resampler.Execute(image)

	return new_image


def whole_flow(dcm_dir: str, re_orient=True, re_origin=True, re_sample=True) -> sitk.Image:
	image = dcm2nii(dcm_dir)

	if re_orient: image = reorient(image)
	if re_origin: image = reorigin(image)
	if re_sample: image = resample(image)

	return image



if __name__ == '__main__':
	work_dir = r'E:\xai-omics\data\20240103-wechat-data\4511_mr_46'
	save_dir = r'E:\BrainSeg\20240103'

	patient_ids = os.listdir(work_dir)
	print(' '.join(patient_ids))

	for i, pid in enumerate(patient_ids):
		# Timing start
		start_time = time.time()

		# Setting path
		patient_path = os.path.join(work_dir, pid)
		patient_save_dir = os.path.join(save_dir, pid)
		if not os.path.exists(patient_save_dir): os.mkdir(patient_save_dir)

		# Get dcm dir
		mr_dir = os.path.join(patient_path, 'mr')
		av45_dir = os.path.join(patient_path, '45')
		av1451_dir = os.path.join(patient_path, '1451')

		ct_pet_av45_name = os.listdir(av45_dir)
		ct_pet_av1451_name = os.listdir(av1451_dir)

		ct_av45_dir = os.path.join(av45_dir, os.listdir(av45_dir)[0])
		pet_av45_dir = os.path.join(av45_dir, os.listdir(av45_dir)[1])
		ct_av1451_dir = os.path.join(av1451_dir, os.listdir(av1451_dir)[0])
		pet_av1451_dir = os.path.join(av1451_dir, os.listdir(av1451_dir)[1])

		# Transform dcm file to nii file and Preprocessing
		mr_nii = whole_flow(mr_dir)
		ct_av45_nii = whole_flow(ct_av45_dir, re_sample=False)
		pet_av45_nii = whole_flow(pet_av45_dir, re_sample=False)
		ct_av1451_nii = whole_flow(ct_av1451_dir, re_sample=False)
		pet_av1451_nii = whole_flow(pet_av1451_dir, re_sample=False)

		# Save the images
		sitk.WriteImage(mr_nii, os.path.join(patient_save_dir, 'raw_mr.nii'))
		# sitk.WriteImage(ct_av45_nii, os.path.join(patient_save_dir, 'raw_45_ct.nii'))
		sitk.WriteImage(pet_av45_nii, os.path.join(patient_save_dir, 'raw_45_pet.nii'))
		# sitk.WriteImage(ct_av1451_nii, os.path.join(patient_save_dir, 'raw_1451_ct.nii'))
		sitk.WriteImage(pet_av1451_nii, os.path.join(patient_save_dir, 'raw_1451_pet.nii'))

		# Timing end
		end_time = time.time()
		elapsed_time = end_time - start_time

		print(f'[{i + 1}/{len(patient_ids)}] {pid} has transformed successfully. Run time: {round(elapsed_time)} s')
