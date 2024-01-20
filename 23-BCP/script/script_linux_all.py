import subprocess
import os
import argparse
from datetime import datetime


def run_shell_script(script, output=False):
	process = subprocess.Popen(script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
	for line in iter(process.stdout.readline, ''):
		if output:
			print(line, end='')
		else:
			if r'Checking or downloading default checkpoints' in line:
				console(r'Start to Segment the MRI...')
			elif r'N4 Bias Correction Parameters' in line:
				console(r'Start to MRI Bias Correction...')
			elif r'To get started, type doc.' in line:
				console(r'Start to write the nii...')
			elif r'www.mathworks.com' in line:
				console(r'Start to Coregister the PET and MRI...')
	process.stdout.close()
	return 

def now_time():
	return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def console(info, level='INFO'):
	print(f'{now_time()} [{level}] {info}')


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='...')

	parser.add_argument('--fs_home', type=str, help='FreeSurfer\'s Home', default=r'/usr/local/freesurfer/7.0.0-dev')
	parser.add_argument('--fast_dir', type=str, help='Fastsufer\'s Dir', default=r'/z3/home/xai_test/brain/FastSurfer')
	parser.add_argument('--env', type=str, help='conda\'s env', default=r'fastsurfer')
	parser.add_argument('--data_dir', type=str, help='data dir', default=r'/z3/home/xai_test/brain/data')
	parser.add_argument('--spm', type=str, help='SPM\'s Dir', default=r'/z3/home/xai_test/brain/spm12')
	parser.add_argument('--pid', nargs='+', type=str, help='pids of the subjects', required=True)

	args = parser.parse_args()

	freesurfer_home = args.fs_home
	fastsufer_dir = args.fast_dir
	conda_env = args.env
	data_dir = args.data_dir
	subjects_dir = os.path.join(data_dir, 'subjects')

	spm_dir = args.spm

	cmd_template = ['#!/bin/sh',
								  f'source activate {conda_env}',
								  f'FREESURFER_HOME={freesurfer_home}',
								  f'SUBJECTS_DIR={subjects_dir}',
									'source $FREESURFER_HOME/SetUpFreeSurfer.sh'
									]

	shell_file = r'/tmp/fs.sh'

	pids = args.pid 

	for pid in pids:
		cmd = cmd_template.copy()
		save_dir = os.path.join(data_dir, 'outputs', pid)
		tmp_dir = os.path.join(save_dir, 'tmp')
		os.makedirs(tmp_dir, exist_ok=True)
		src_dir = os.path.join(data_dir, 'raw', pid)

		# Segmentation
		
		cmd.append(f'{fastsufer_dir}/run_fastsurfer.sh --sd $SUBJECTS_DIR --sid {pid} --t1 {src_dir}/raw_mr.nii --seg_only --device cuda --ignore_fs_version --threads 6 --parallel')

		# Format transformation
		  
		cmd.append(f'mri_convert $SUBJECTS_DIR/{pid}/mri/orig.mgz {save_dir}/fastsurfer_mr.nii')
		cmd.append(f'mri_convert $SUBJECTS_DIR/{pid}/mri/aparc.DKTatlas+aseg.deep.mgz {save_dir}/dk_mask_fastsurfer.nii')

		# Coregister
		# av45 and av1451
		cmd.append(f'matlab -nodisplay -nosplash -nodesktop -r "addpath(\'{spm_dir}\');coregister(\'{src_dir}/raw_45_pet.nii\',\'{save_dir}/fastsurfer_mr.nii\');exit;"')
		cmd.append(f'matlab -nodisplay -nosplash -nodesktop -r "addpath(\'{spm_dir}\');coregister(\'{src_dir}/raw_1451_pet.nii\',\'{save_dir}/fastsurfer_mr.nii\');exit;"')

		cmd.append(f'mv {src_dir}/rraw_45_pet.nii {save_dir}/cor_45_pet.nii')
		cmd.append(f'mv {src_dir}/rraw_1451_pet.nii {save_dir}/cor_1451_pet.nii')

		with open(shell_file, 'w') as f:
			f.write('\n'.join(cmd))

		# print(cmd)

		console(f'Start to Process pid {pid}')
		run_shell_script(f'bash {shell_file}', output=True)
		console(f'pid {pid} done, the output is saved in {save_dir}')

		# break
		

