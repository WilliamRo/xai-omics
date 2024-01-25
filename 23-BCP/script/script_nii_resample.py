import os
import SimpleITK as sitk
import numpy as np



def resampleVolume(outspacing, vol):
	inputsize = vol.GetSize()
	inputspacing = vol.GetSpacing()

	transform = sitk.Transform()
	transform.SetIdentity()

	outsize = [
		int(inputsize[i] * inputspacing[i] / outspacing[i] + 0.5)
		for i in range(3)]

	resampler = sitk.ResampleImageFilter()
	resampler.SetTransform(transform)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetOutputOrigin(vol.GetOrigin())
	resampler.SetOutputSpacing(outspacing)
	resampler.SetOutputDirection(vol.GetDirection())

	# new_out_size = [int(2 ** np.ceil(np.log2(size))) for size in outsize]
	# resampler.SetSize(new_out_size)
	resampler.SetSize(outsize)
	newvol = resampler.Execute(vol)

	return newvol



if __name__ == '__main__':
	work_dir = r'E:\BrainSeg\fastsurfer'
	pids = os.listdir(work_dir)

	for p in pids:
		mr_path = os.path.join(work_dir, p, 'raw_mr.nii')
		save_path = os.path.join(work_dir, p, 'resample_mr.nii')

		mr = sitk.Image(sitk.ReadImage(mr_path))

		if mr.GetSpacing() == [1, 1, 1]:
			new_mr = mr
		else:
			new_mr = resampleVolume([1, 1, 1], mr)

		sitk.WriteImage(new_mr, save_path)
