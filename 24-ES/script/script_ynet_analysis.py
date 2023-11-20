import os
import openpyxl

from xomics import MedicalImage
from tqdm import tqdm

def save_summary_excel(input_dir, excel_path):
  model_name = os.listdir(input_dir)
  model_name = [m for m in model_name if '.xlsx' not in m]

  total_key = []
  for m in tqdm(model_name, desc='Model'):
    model_dir = os.path.join(dir, m)
    if not os.path.isdir(model_dir): continue

    mi_files = [f for f in os.listdir(model_dir) if '.mi' in f]
    mi_keys = [MedicalImage.load(os.path.join(model_dir, f)).key
               for f in tqdm(mi_files, desc='mi file')]
    total_key.append(mi_keys)

  assert all(len(l) == len(total_key[0]) for l in total_key)

  workbook = openpyxl.Workbook()
  sheet = workbook.active

  title = ['pid'] + [m for m in model_name]
  sheet.append(title)

  for i in range(len(total_key[0])):
    pid = [total_key[0][i].split('---')[0]]
    acc = [float(tk[i].split('Acc:')[-1]) for tk in total_key]
    sheet.append(pid + acc)

  workbook.save(excel_path)
  print(f'{os.path.abspath(excel_path)} saved successfully!!!')



if __name__ == '__main__':
  dir = r'../../data/02-PET-CT-Y1/results/03-ynet/mi'
  excel_path = os.path.join(dir, 'acc summary_1.xlsx')

  if not os.path.exists(excel_path):
    save_summary_excel(dir, excel_path)
  else:
    assert TypeError(f'{excel_path} has existed!!!' )

  print()






