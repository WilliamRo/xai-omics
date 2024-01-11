import os, csv

data_dir = '../../../data/02-RLD'


def find_list(a, key):
  for i in a:
    if i.endswith('.pkl'):continue
    if key in i:
      return i



if __name__ == '__main__':
  csvfile = [['pid', 'CT', '20S', '30G', '40S', '60G', '120S', '240G', '240S','CT_seg']]
  csv_path = os.path.join(data_dir, 'rld_data.csv')

  root_dir = r'D:/projects/xai-omics/data/02-RLD'
  # root_dir = r'/z3/home/xai_test/jimmy/pytorch/data/02-RLD'
  for dirs in os.listdir(data_dir):
    if not dirs.startswith('sub'): continue
    files = sorted(os.listdir(os.path.join(data_dir, dirs)))
    ids = files[0].split('_')[-1].split('.')[0]
    path = os.path.join(dirs, files[0])
    # CT 20S 30G 40S 60G 120S 240G 240S
    row = [ids]
    # print(dirs)
    func = lambda x: os.path.join(root_dir, dirs, find_list(files, x)).replace('\\', '/')
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

  with open(csv_path, 'w', newline='\n') as f:
    writer = csv.writer(f)
    writer.writerows(csvfile)

