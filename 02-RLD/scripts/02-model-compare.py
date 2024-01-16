import os, re
import numpy as np

from roma import console


def get_save_dir(root_dir):
  models_dir = list_folder(root_dir, r'^[0-9]{2}_[a-z]+')
  console.show_status(f'Models:')
  for i, model in enumerate(models_dir):
    console.supplement(f'[{i}] {model}', level=2)
  console.show_status(f'Choose your model:')
  model = models_dir[int(input())]

  console.show_status(f'Choose your checkpoint:')
  checkpoints = os.listdir(os.path.join(root_dir, model, 'checkpoints'))
  for i, checkpoint in enumerate(checkpoints):
    console.supplement(f'[{i}] {checkpoint}', level=2)
  checkpoint = checkpoints[int(input())]

  save_dir = os.path.join(root_dir, model, 'checkpoints', checkpoint, 'saves')
  return save_dir


def list_folder(folder_path, pattern):
  matched_files = []
  for files in os.listdir(folder_path):
      if re.match(pattern, files):
        matched_files.append(files)
  return matched_files


def load_metrics(save_dir):
  path = os.path.join(save_dir)
  metric = np.loadtxt(path, delimiter=',', dtype=np.str_)
  name = metric[0, 1:]
  metric = metric[1:, 1:].astype(np.float64)



if __name__ == '__main__':
  root_dir = f'../'
  save_dir = get_save_dir(root_dir)




  pass

