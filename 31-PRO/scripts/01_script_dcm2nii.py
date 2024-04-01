import os
import SimpleITK as sitk
import numpy as np
import pydicom
import time

from xomics.data_io.utils.preprocess import calc_SUV



def dcm2nii(dcm_dir, suv_correction=False) -> sitk.Image:
	reader = sitk.ImageSeriesReader()
	dcm_names = reader.GetGDCMSeriesFileNames(dcm_dir)
	reader.SetFileNames(dcm_names)
	image = reader.Execute()

	if suv_correction:
		array = sitk.GetArrayFromImage(image)
		dcm = pydicom.dcmread(os.path.join(dcm_dir, os.listdir(dcm_dir)[1]))
		ST = dcm.SeriesTime
		AT = dcm.AcquisitionTime
		PW = dcm.PatientWeight
		RIS = dcm.RadiopharmaceuticalInformationSequence[0]
		RST = str(RIS['RadiopharmaceuticalStartTime'].value)
		RTD = str(RIS['RadionuclideTotalDose'].value)
		RHL = str(RIS['RadionuclideHalfLife'].value)
		RS = dcm.RescaleSlope
		RI = dcm.RescaleIntercept
		dcm_tag = {
			'ST': ST,
			'AT': AT,
			'PW': PW,
			'RST': RST,
			'RTD': RTD,
			'RHL': RHL,
			'RS': RS,
			'RI': RI
		}
		array = calc_SUV(array, tags=dcm_tag, norm=False)
		new_image = sitk.GetImageFromArray(array)
		new_image.CopyInformation(image)
		return new_image

	return image


def resample_image_with_reference(input_image, reference_image) -> sitk.Image:
	reference_size = reference_image.GetSize()
	reference_spacing = reference_image.GetSpacing()
	reference_origin = reference_image.GetOrigin()
	reference_direction = reference_image.GetDirection()

	interpolator = sitk.sitkLinear

	resampled_image = sitk.Resample(
		input_image, size=reference_size, outputSpacing=reference_spacing,
		outputOrigin=reference_origin, outputDirection=reference_direction,
		interpolator=interpolator)

	return resampled_image



if __name__ == '__main__':
	'''
	Prostate data preprocessing scripts
	Resample PET image with reference (CT image)
	Produce nii file from dcm files
	'''
	# Input
	work_dir = r'E:\xai-omics\data\31-Prostate\raw_data'
	image_dir = os.path.join(work_dir, 'ct_pet')
	patient_ids = os.listdir(image_dir)[-41:]

	# Output
	save_dir = os.path.join(work_dir, 'nii')

	# Parameter
	spacing = None
	time_total_start = time.time()

	for p_index, p in enumerate(patient_ids):
		p_time_start = time.time()
		files = os.listdir(os.path.join(image_dir, p))
		file0 = os.path.join(image_dir, p, files[0])
		file1 = os.path.join(image_dir, p, files[1])
		ct_dcm_dir = os.path.join(image_dir, p, 'ct')
		pet_dcm_dir = os.path.join(image_dir, p, 'pet')
		os.rename(file0, ct_dcm_dir)
		os.rename(file1, pet_dcm_dir)

		ct_image = dcm2nii(ct_dcm_dir)
		pet_image = dcm2nii(pet_dcm_dir, True)

		save_path = os.path.join(save_dir, p)
		if not os.path.exists(save_path): os.mkdir(save_path)

		resample_pet_image = resample_image_with_reference(pet_image, ct_image)

		new_pet_array = sitk.GetArrayFromImage(resample_pet_image).astype(np.float32)
		assert np.all(np.isfinite(new_pet_array))

		new_pet_image = sitk.GetImageFromArray(new_pet_array)
		new_pet_image.CopyInformation(resample_pet_image)

		sitk.WriteImage(ct_image, os.path.join(save_path, 'ct.nii.gz'))
		sitk.WriteImage(new_pet_image, os.path.join(save_path, 'pet.nii.gz'))


		p_time_end = time.time()
		p_time = p_time_end - p_time_start
		time_total = p_time_end - time_total_start
		print(
			f"[{p_index + 1}/{len(patient_ids)}] PIDS:{p}\t\t\t\t\t\t"
			f"Consuming Time: {p_time: .2f} s\t\t\t"
			f"total: {time_total: .2f} s")
