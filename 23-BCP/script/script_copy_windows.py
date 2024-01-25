import os
import shutil
import subprocess


def copy_file(source_path, destination_path):
	try:
		shutil.copy2(source_path, destination_path)
		print(f"File copied from {source_path} to {destination_path}")
	except Exception as e:
		print(f"Error copying file: {e}")

def move_file(source_path, destination_path):
	try:
		shutil.move(source_path, destination_path)
		print(f"文件 {source_path} 移动到 {destination_path} 成功。")
	except FileNotFoundError:
		print(f"找不到文件 {source_path}。")
	except Exception as e:
		print(f"移动文件时出错：{e}")


def move_files_by_extension(source_folder, destination_folder, extension=".nii"):
	try:
		if not os.path.exists(destination_folder):
			os.makedirs(destination_folder)

		for filename in os.listdir(source_folder):
			if filename.endswith(extension):
				source_path = os.path.join(source_folder, filename)
				destination_path = os.path.join(destination_folder, filename)

				shutil.move(source_path, destination_path)
				print(f"文件 {filename} 移动到 {destination_folder} 成功。")

	except Exception as e:
		print(f"移动文件时出错：{e}")



if __name__ == '__main__':
	destination_dir = r'E:\BrainSeg\20240103'
	source_dir = r'E:\BrainSeg\20240103\outputs'

	patient_ids = [f for f in os.listdir(destination_dir) if 'output' not in f]

	for pid in patient_ids:
		source_path = os.path.join(source_dir, pid)
		destination_path = os.path.join(destination_dir, pid)
		move_files_by_extension(source_path, destination_path)

