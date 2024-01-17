import os, re

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from roma import console
from utils.statistics import add_title_row_col, regions, hist_joint
from xomics.data_io.utils.metrics_calc import get_metrics
from xomics.objects.general_mi import GeneralMI


def get_save_dir(root_dir):
  models_dir = list_folder(root_dir, r'^[0-9]{2}_[a-z]+')
  save_dirs = {}
  for i, model in enumerate(models_dir):
    save_dirs[model] = []
    checkpoints = os.listdir(os.path.join(root_dir, model, 'checkpoints'))
    for j, checkpoint in enumerate(checkpoints):
      save_dir = os.path.join(root_dir, model, 'checkpoints', checkpoint, 'saves')
      if not os.path.exists(save_dir):
        continue
      save_dirs[model].append(save_dir.replace('\\', '/'))
  return save_dirs


def get_checkpoint_dir(root_dir, model):
  checkpoints = os.listdir(os.path.join(root_dir, model, 'checkpoints'))
  for i, checkpoint in enumerate(checkpoints):
    console.supplement(f'[{i}] {checkpoint}', level=2)



def list_folder(folder_path, pattern):
  matched_files = []
  for files in os.listdir(folder_path):
      if re.match(pattern, files):
        matched_files.append(files)
  return matched_files


def load_imgs(compare_dir, raw_dir, raw, models, fake=True):
  img_list = []
  pid = []
  for i, dirpath in enumerate(compare_dir):
    console.print_progress(i, len(compare_dir))
    files = list_folder(dirpath, r'^.*\.nii.gz')
    if len(pid) == 0:
      pid = [file.split('-')[1] for file in files]
    if fake:
      img_list.append([])
      continue
    img = [GeneralMI.load_img(os.path.join(dirpath, file), True)
           for file in files]
    img_list.append(img)
  console.clear_line()

  imgs = dict(zip(models, img_list))

  for i, name in enumerate(raw):
    console.print_progress(i, len(raw))
    files = list_folder(raw_dir, f'^.*-{name}\\.nii.gz')
    files = sorted(files, key=lambda x: pid.index(x.split('-')[0]))
    if fake:
      imgs[name] = [None]*5
      continue
    imgs[name] = [GeneralMI.load_img(os.path.join(raw_dir, file), True)
                  for file in files]
  console.clear_line()
  return imgs, np.array(pid)


def get_metrics_table(imgs, pid, metrics, models, tmp_dir, raw):
  status_path = os.path.join(tmp_dir, 'metrics_done')
  target = imgs['full']
  metric_arr = np.zeros(
    (len(imgs.items()) - len(raw) + 1, len(target), len(metrics)))

  if os.path.exists(status_path):
    files = list_folder(tmp_dir, r'^.*_metrics.csv')
    for i, file in enumerate(files):
      metric_arr[i] = np.loadtxt(os.path.join(tmp_dir, file, ), dtype=np.str_,
                                 delimiter=',')[1:, 1:].astype(np.float64)
  else:
    max_suv = [np.max(imgs['low'][i]) for i in range(len(target))]
    j = 0
    for key, feature in imgs.items():
      if key in ['full', 'seg']: continue
      for i in range(len(target)):
        m_result = get_metrics(target[i] / max_suv[i], feature[i] / max_suv[i],
                               metrics)
        for t, k in enumerate(m_result):
          metric_arr[j, i, t] = m_result[k]
      j += 1
    for i, arr in enumerate(metric_arr):
      save_arr = add_title_row_col(arr, [""] + metrics, pid)
      tmp_path = os.path.join(tmp_dir, f'{models[i]}_metrics.csv')
      np.savetxt(tmp_path, save_arr, delimiter=',', fmt="%s")
    os.mkdir(status_path)
  return metric_arr


def get_suv_data(imgs, pid, roi, roi_label, models, tmp_dir):
  status_path = os.path.join(tmp_dir, 'suv_done')
  targets = imgs['full']
  suv_max = np.zeros((len(models), len(targets), len(roi)))
  suv_mean = np.zeros((len(models), len(targets), len(roi)))

  if os.path.exists(status_path):
    region_pd = joblib.load(os.path.join(tmp_dir, 'region_pd'))
    region_data = joblib.load(os.path.join(tmp_dir, 'region_data'))
    for i in range(len(models)):
      suv_max[i] = np.loadtxt(os.path.join(tmp_dir, f'{models[i]}_suv_max.csv'),
                              dtype=np.str_, delimiter=',')[1:, 1:].astype(np.float64)
      suv_mean[i] = np.loadtxt(os.path.join(tmp_dir, f'{models[i]}_suv_mean.csv'),
                              dtype=np.str_, delimiter=',')[1:, 1:].astype(np.float64)
  else:
    region_data = []
    seg = imgs['seg']
    for k, key in enumerate(models):
      res = imgs[key]
      for i, img in enumerate(res):
        for t, r in enumerate(roi):
          if isinstance(r, list):
            onehot = np.isin(seg[i], r)
          else:
            onehot = seg[i] == r
          roi_img = img[onehot]
          s_max, s_mean = np.max(roi_img), np.mean(roi_img)
          suv_max[k, i, t] = s_max
          suv_mean[k, i, t] = s_mean
          region_data.append(roi_img)
    for i in range(len(models)):
      max_path = os.path.join(tmp_dir, f'{models[i]}_suv_max.csv')
      mean_path = os.path.join(tmp_dir, f'{models[i]}_suv_mean.csv')
      max_arr = add_title_row_col(suv_max[i], [""] + roi_label, pid)
      mean_arr = add_title_row_col(suv_mean[i], [""] + roi_label, pid)
      np.savetxt(max_path, max_arr, delimiter=',', fmt="%s")
      np.savetxt(mean_path, mean_arr, delimiter=',', fmt="%s")
    os.mkdir(status_path)

    model = np.repeat(models, 20)
    rois = np.tile(roi_label, 25)
    pid = np.tile(np.repeat(pid, 4), 5)
    pd_data = pd.DataFrame({
      'model': model,
      'roi': rois,
      'pid': pid,
      'data': region_data,
    })
    joblib.dump(region_data, os.path.join(tmp_dir, 'region_data'))
    region_pd = pd.DataFrame(columns=['pixel', 'pid', 'roi', 'model'])
    for _, row in pd_data.iterrows():
      region_pd = pd.concat([region_pd, pd.DataFrame({
        'pixel': row['data'],
        'pid': np.tile([row['pid']], len(row['data'])),
        'roi': np.tile([row['roi']], len(row['data'])),
        'model': np.tile([row['model']], len(row['data'])),
      })], ignore_index=True)
    joblib.dump(region_pd, os.path.join(tmp_dir, 'region_pd'))

  return suv_max, suv_mean, region_pd, region_data


def draw_bar_with_err(ax, x, y, y_err, width, label):
  ax.bar(x, y, width, label=label)
  ax.errorbar(x, y, yerr=y_err, fmt='.', color='red', ecolor='black', capsize=6)


if __name__ == '__main__':
  root_dir = f'../'
  tmp_dir = f'./tmp'
  save_dir = get_save_dir(root_dir)
  compare_dir = []
  raw_dir = f'../06_puadap/raw_data'
  raw = ['low', 'full', 'seg']
  models = ['03_uadap', '05_upl', '06_puadap']
  select_check = [0, 1, 1]

  # data load
  for i, model in enumerate(models):
    console.supplement(f'[{i}] {model}:', level=1)
    for j, checkpoints in enumerate(save_dir[model]):
      console.supplement(f'[{j}] {checkpoints.split("/")[-2]}', level=2)
      if select_check[i] == j:
        compare_dir.append(checkpoints)
  imgs, pid = load_imgs(compare_dir, raw_dir, raw, models, fake=False)

  # Metrics
  metrics = ['SSIM', 'NRMSE', 'RELA', 'PSNR']

  metric_arr = get_metrics_table(imgs, pid, metrics,
                                 models+raw[:1], tmp_dir, raw)

  # SUV
  roi = [5, [10, 11], [12, 13, 14], 51]
  roi_label = ['liver', 'left lung', 'right lung', 'heart']

  suv_max, suv_mean, region_pd, region_data = get_suv_data(imgs, pid, roi, roi_label,
                                                           models+raw[:-1], tmp_dir)

  # Draw
  width = 0.2
  fig, axs = plt.subplots(2, 2, figsize=(12, 12))
  func = ['metric', 'suv_dis', 'suv', 'hist'][-1:]
  # Metric
  if 'metric' in func:
    metric_x = np.arange(len(metrics))
    mean = np.mean(metric_arr, axis=1)
    std = np.std(metric_arr, axis=1)
    mean[[0, -1]] = mean[[-1, 0]]
    std[[0, -1]] = std[[-1, 0]]
    labels = ['Low'] + models
    psnr_ax = axs[0, 0].twinx()
    x = metric_x
    for i, label in enumerate(labels):
      draw_bar_with_err(axs[0, 0], x[0], mean[i][0], std[i][0], width, label)
      draw_bar_with_err(psnr_ax, x[1], mean[i][-1], std[i][-1], width, label)
      draw_bar_with_err(axs[0, 1], x[2:], mean[i][1:3], std[i][1:3], width, label)
      x = x + 1 / len(metrics)
    axs[0, 0].set_ylim(0.98, 1)
    psnr_ax.set_ylim(40, 70)
    axs[0, 1].set_ylim(0.1, 0.6)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 0].set_xticks([0.325, 1.325], metrics[:1]+metrics[-1:])
    axs[0, 1].set_xticks([2.325, 3.325], metrics[1:3])
    axs[0, 0].set_title('Metrics')
    axs[0, 1].set_title('Loss')

  # SUV
  if 'suv_dis' in func:
    data = region_pd[
      (region_pd['model'] == '05_upl') |
      (region_pd['model'] == 'full') |
      (region_pd['model'] == 'low')
      ]
    sns.violinplot(y='pixel', x='roi', hue='model', data=data, ax=axs[1, 0], inner='quartile')

    data = region_pd[region_pd['roi'] == 'liver']
    sns.violinplot(y='pixel', x='model', data=data, ax=axs[1, 1], inner='quartile')

    axs[1, 0].set_title('the SUV Distribution of ROI (U-Net)')
    axs[1, 1].set_title('the SUV Distribution of liver')
    axs[1, 0].set_ylim(0, 5)
    axs[1, 1].set_ylim(0, 5)

  if 'suv' in func:
    mod = raw[0:1] + models + raw[-2:-1]
    model_x = np.arange(len(mod))
    roi_x = np.arange(len(roi_label))
    x1 = model_x
    x2 = roi_x

    s_max = np.mean(suv_max, axis=1)
    s_mean = np.mean(suv_mean, axis=1)
    std_max = np.std(suv_max, axis=1)
    std_mean = np.std(suv_mean, axis=1)

    for i in range(len(roi_label)):
      draw_bar_with_err(axs[0, 0], x1, s_mean[:, i], std_mean[:, i],
                        width, roi_label[i])
      x1 = x1 + 1 / len(roi_label)

    for i in range(len(mod)):
      draw_bar_with_err(axs[0, 1], x2, s_mean[i], std_mean[i],
                        width/1.5, mod[i])
      x2 = x2 + 1 / len(mod)
    axs[0, 0].set_xticks(model_x, mod)
    axs[0, 0].legend()
    axs[0, 1].set_xticks(roi_x, roi_label)
    axs[0, 1].legend()
    axs[0, 0].set_title('$SUV_{mean}$')
    axs[0, 1].set_title('$SUV_{mean}$')


  if 'hist' in func:
    hist_joint(fig, axs[0, 0], imgs['full'][0], imgs['low'][0],
               '240s Gated', '30s Gated', -3, 3)
    axs[0, 0].set_title(f'Joint Voxel Histogram - Log Scale '
                        f'(NRMSE = {metric_arr[-1][0][2]:.2f})')
    hist_joint(fig, axs[0, 1], imgs['full'][0], imgs['05_upl'][0],
               '240s Gated', '05_upl', -3, 3)
    axs[0, 1].set_title(f'Joint Voxel Histogram - Log Scale '
                        f'(NRMSE = {metric_arr[1][0][2]:.2f})')


  plt.show()


  pass

