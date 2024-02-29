import csv
import os
import re


def find_list(a, key):
  for i in a:
    if i.endswith('.pkl'):continue
    if key in i:
      return i


def list_folder(folder_path, pattern):
  matched_files = []
  for files in os.listdir(folder_path):
      if re.match(pattern, files):
        matched_files.append(files)
  return matched_files


match_dict = {
  'CT': 'CT_WB',
  '20S': 'STATIC_20S',
  '30G': 'GATED_30S',
  '40S': 'STATIC_40S',
  '90G': 'GATED_90S',
  '60G-1': 'GATED_60S_',
  '60G-2': 'GATED_60S-GROUP2',
  '60G-3': 'GATED_60S-GROUP3',
  '120S': 'STATIC_120S',
  '120G': 'GATED_120S',
  '240G': 'GATED_240S',
  '240S': 'STATIC_240S',
  'CT_seg': 'CT_seg',
}


def get_key_file(lst, key):
  for i in lst:
    if key in i:
      return i


def gen_path_list(imgs_dir, pid, patient_path):
  path_list = [pid] + ['']*len(match_dict)
  for i, v in enumerate(match_dict.values()):
    img = get_key_file(imgs_dir, v)
    if not img:
      continue
    img_path = os.path.join(patient_path, img).replace('\\', '/')
    path_list[i+1] = img_path
  return path_list

def old_method(data_dir, csvfile):
  for dirs in os.listdir(data_dir):
    if not dirs.startswith('sub'): continue
    files = sorted(os.listdir(os.path.join(data_dir, dirs)))
    ids = files[0].split('_')[-1].split('.')[0]
    path = os.path.join(dirs, files[0])
    # CT 20S 30G 40S 60G 120S 240G 240S
    row = [ids]
    # print(dirs)
    func = lambda x: os.path.join(data_dir, dirs, find_list(files, x)).replace('\\', '/')
    row.append(func('CT_WB'))
    row.append(func('PET_WB_20'))
    row.append(func('PET_WB_30'))
    row.append(func('PET_WB_40'))
    row.append(func('PET_WB_60'))
    row.append(func('PET_WB_120'))
    row.append(func('PET_WB_240S_GATED'))
    row.append(func('PET_WB_240S_STATIC'))
    row.append(func('CT_seg'))
    # print(row)
    csvfile.append(row)


def gen_csv(data_dir, csvfile):
  if os.path.exists(csv_path):
    os.remove(csv_path)
  patients_dir = sorted(os.listdir(data_dir))
  for patient_dir in patients_dir:
    patient_path = os.path.join(data_dir, patient_dir)
    if os.path.isfile(patient_path):
      continue
    pid = patient_dir
    imgs = list_folder(patient_path, r'.*\.nii\.gz')
    path_list = gen_path_list(imgs, pid, patient_path)
    csvfile.append(path_list)

  with open(csv_path, 'w', newline='\n') as f:
    writer = csv.writer(f)
    writer.writerows(csvfile)



if __name__ == '__main__':
  csvfile = [['pid']+list(match_dict.keys())]

  data_dir = 'D:/projects/xai-omics/data/02-RLD-0226'
  # data_dir = r'\\192.168.5.99/xai/xai-omics/data/02-RLD-0226'
  # data_dir = r'/z3/home/xai_test/xai-omics/data/02-RLD'
  csv_path = os.path.join(data_dir, 'rld_data.csv')

  # old_method(data_dir, csvfile)

  gen_csv(data_dir, csvfile)
  pass



