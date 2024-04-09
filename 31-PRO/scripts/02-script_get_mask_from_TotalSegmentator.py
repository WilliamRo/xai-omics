import subprocess
import os
import time



if __name__ == '__main__':
	'''
	Running in Windows
	Get Prostate and urinary_bladder mask by TotalSegmentator
	Usage:
		Open cmd
		cd E:\\xai-omics\data\\31-Prostate
		conda activate ct_seg
		python ..\\..\\31-PRO\\scripts\\22-script_get_mask_from_TotalSegmentator.py
	'''
	# Input
	work_dir = r'../../data/31-Prostate'
	patient_ids = os.listdir(os.path.join(work_dir, 'mask_and_image'))
	patient_ids = sorted(patient_ids, key=lambda x: int(x.split('_')[0]))[-41:]
	print(patient_ids)

	# Output

	# Parameter
	time_total_start = time.time()

	for p_index, p in enumerate(patient_ids):
		p_time_start = time.time()

		ct_file_path = os.path.abspath(
			os.path.join(work_dir, 'mask_and_image', p, 'ct.nii.gz'))

		save_dir = os.path.join(work_dir, 'mask_all', p)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_path = os.path.abspath(os.path.join(save_dir, 'mask.nii.gz'))

		cmd_seg = f'TotalSegmentator -i {ct_file_path} -o {save_path} --ml'

		subprocess.run(cmd_seg, shell=True)

		p_time_end = time.time()
		p_time = p_time_end - p_time_start
		time_total = p_time_end - time_total_start
		print(
			f"[{p_index + 1}/{len(patient_ids)}] PIDS:{p}\t\t\t\t\t\t"
			f"Consuming Time: {p_time: .2f} s\t\t\t"
			f"total: {time_total: .2f} s")
