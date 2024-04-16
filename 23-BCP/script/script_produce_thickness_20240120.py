import os
import openpyxl

def get_ctx_num():
	braak_num = [0 for _ in range(31)]

	number_left = [
		1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
		1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026,
		1027, 1028, 1029, 1030, 1031, 1034, 1035]
	number_right = [n + 1000 for n in number_left]

	atlas_dict = get_region_reference_num_from_DK_atlas()

	for n_index, number in enumerate(number_left):
		for k_index, k in enumerate(list(atlas_dict.keys())[:6]):
			if number in atlas_dict[k]:
				braak_num[n_index] = k_index + 1
				continue

	return number_left + number_right, braak_num + braak_num


def get_region_reference_num_from_DK_atlas():
  # whole cerebellum: [7, 46, 8, 47]
  # cerebellum without graymatter: [8, 47]
  atlas_dict = {}

  region_num_braak1_left = [1006]
  region_num_braak1_right = [2006]

  region_num_braak2_left = [17]
  region_num_braak2_right = [53]

  region_num_braak3_left = [1016, 1007, 1013, 18]
  region_num_braak3_right = [2016, 2007, 2013, 54]

  region_num_braak4_left = [1015, 1002, 1026, 1023, 1010, 1035, 1009, 1033]
  region_num_braak4_right = [2015, 2002, 2026, 2023, 2010, 2035, 2009, 2033]

  region_num_braak5_left = [1012, 1014, 1032, 1003, 1027, 1018, 1019, 1020, 1011, 1031, 1008, 1030, 1029, 1025, 1001, 1034, 1028]
  region_num_braak5_right = [2012, 2014, 2032, 2003, 2027, 2018, 2019, 2020, 2011, 2031, 2008, 2030, 2029, 2025, 2001, 2034, 2028]

  region_num_braak6_left = [1021, 1022, 1005, 1024, 1017]
  region_num_braak6_right = [2021, 2022, 2005, 2024, 2017]


  atlas_dict['b1_left'] = region_num_braak1_left
  atlas_dict['b2_left'] = region_num_braak2_left
  atlas_dict['b3_left'] = region_num_braak3_left
  atlas_dict['b4_left'] = region_num_braak4_left
  atlas_dict['b5_left'] = region_num_braak5_left
  atlas_dict['b6_left'] = region_num_braak6_left

  atlas_dict['b1_right'] = region_num_braak1_right
  atlas_dict['b2_right'] = region_num_braak2_right
  atlas_dict['b3_right'] = region_num_braak3_right
  atlas_dict['b4_right'] = region_num_braak4_right
  atlas_dict['b5_right'] = region_num_braak5_right
  atlas_dict['b6_right'] = region_num_braak6_right

  return atlas_dict


def get_cortical_thickness_dict(input_file):
	result_dict = {}
	with open(input_file, 'r') as f:
		lines = f.read().split('\n')

	for line in lines:
		items = line.split()
		result_dict[items[0]] = [items[1] + ' ± ' + items[2], items[3] + ' ± ' + items[4]]

	return result_dict



if __name__ == '__main__':
	work_dir = r'E:\xai-omics\data\30-Brain-SQA\2024-01-20-77\outputs_20240120_thickness'
	pids = [f for f in os.listdir(work_dir)
					if os.path.isdir(os.path.join(work_dir, f)) and f != 'mi']
	cortical_thickness_file = r'E:\xai-omics\data\30-Brain-SQA\2024-01-20-77\cortical_thickness.txt'

	save_dir = r'E:\xai-omics\data\30-Brain-SQA\2024-01-20-77\results'
	excel_name = '2024-01-20-thickness.xlsx'
	save_path = os.path.join(save_dir, excel_name)

	cortical_thickness_dict = get_cortical_thickness_dict(cortical_thickness_file)

	excel_data = {}
	title = []
	for p_index, p in enumerate(pids):
		excel_data[p] = [p]
		file_names = ['lh.aparc.stats', 'rh.aparc.stats']

		for f in file_names:
			status_path = os.path.join(work_dir, p, f)

			if not os.path.exists(status_path):
				print(f'[ERROR] {f} of {p} missed!')
				excel_data[p] = excel_data[p] + ['' for _ in range(31)]
				continue

			with open(status_path, 'r') as f:
				raw_data = f.read()

			raw_data = [c for c in raw_data.split('\n') if len(c) > 0 and c[0] != '#']
			raw_data = [i.split() for i in raw_data]
			assert all([len(l) == 10 for l in raw_data])
			ctx_name = list(zip(*raw_data))[0]
			mean = list(zip(*raw_data))[4]
			std = list(zip(*raw_data))[5]

			excel_data[p] = excel_data[p] + [f'{m} ± {s}' for m, s in zip(mean, std)]

			if len(title) == 0:
				title = ['lh_' + n for n in ctx_name]
			elif len(title) == 31:
				title.extend(['rh_' + n for n in ctx_name])
			else:
				pass

	# Save as excel
	workbook = openpyxl.Workbook()
	sheet = workbook.active
	sheet.append(['PIDS'] + title + ['lh_cortical_thickness', 'rh_cortical_thickness'])

	ctx_number, braak_number = get_ctx_num()
	sheet.append(['Number'] + ctx_number)
	sheet.append(['Braak'] + braak_number)

	for key in excel_data:
		if key in cortical_thickness_dict.keys():
			sheet.append(excel_data[key] + cortical_thickness_dict[key])
		else:
			sheet.append(excel_data[key])

	workbook.save(save_path)
	print(f'Excel file saved to {save_path}')
