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


def load_imgs(compare_dir, raw_dir, raw, models, raw_name, fake=True):
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

  for i, name in enumerate(raw_name):
    console.print_progress(i, len(raw))
    files = list_folder(raw_dir, f'^.*-{name}\\.nii.gz')
    files = sorted(files, key=lambda x: pid.index(x.split('-')[0]))
    if fake:
      imgs[raw[i]] = [None]*5
      continue
    imgs[raw[i]] = [GeneralMI.load_img(os.path.join(raw_dir, file), True)
                  for file in files]
  console.clear_line()
  return imgs, np.array(pid)


def get_metrics_table(imgs, pid, metrics, models, tmp_dir, raw):
  status_path = os.path.join(tmp_dir, 'metrics_done')
  target = imgs['full']
  metric_arr = np.zeros(
    (len(imgs.items()) - len(raw) + 1, len(target), len(metrics)))

  if os.path.exists(status_path):
    files = list_folder(tmp_dir, r'^([0-9]|low).*_metrics.csv')
    for i, file in enumerate(files):
      item = models.index(file[:-12])
      metric_arr[item] = np.loadtxt(os.path.join(tmp_dir, file, ), dtype=np.str_,
                                 delimiter=',')[1:, 1:].astype(np.float64)
  else:
    # max_suv = [np.max(imgs['low'][i]) for i in range(len(target))]
    for key, feature in imgs.items():
      if key in ['full', 'seg']: continue
      for i in range(len(target)):
        m_result = get_metrics(target[i], feature[i], metrics,
                               data_range=np.max([target[i], feature[i]]))
        j = models.index(key)
        for t, k in enumerate(m_result):
          metric_arr[j, i, t] = m_result[k]

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
  data_path = os.path.join(tmp_dir, 'region_data')

  if os.path.exists(status_path):
    region_pd = joblib.load(os.path.join(tmp_dir, 'region_pd'))
    region_data = joblib.load(data_path)
    for i in range(len(models)):
      suv_max[i] = np.loadtxt(os.path.join(tmp_dir, f'{models[i]}_suv_max.csv'),
                              dtype=np.str_, delimiter=',')[1:, 1:].astype(np.float64)
      suv_mean[i] = np.loadtxt(os.path.join(tmp_dir, f'{models[i]}_suv_mean.csv'),
                              dtype=np.str_, delimiter=',')[1:, 1:].astype(np.float64)
  else:
    if not os.path.exists(data_path):
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

      joblib.dump(region_data, data_path)
    else:
      region_data = joblib.load(data_path)

    model = np.repeat(models, 20)
    rois = np.tile(roi_label, 25)
    pid = np.tile(np.repeat(pid, 4), 5)
    pd_data = pd.DataFrame({
      'model': model,
      'roi': rois,
      'pid': pid,
      'data': region_data,
    })

    region_pd = pd.DataFrame(columns=['pixel', 'pid', 'roi', 'model'])
    for _, row in pd_data.iterrows():
      region_pd = pd.concat([region_pd, pd.DataFrame({
        'pixel': row['data'],
        'pid': np.tile([row['pid']], len(row['data'])),
        'roi': np.tile([row['roi']], len(row['data'])),
        'model': np.tile([row['model']], len(row['data'])),
      })], ignore_index=True)

    joblib.dump(region_pd, os.path.join(tmp_dir, 'region_pd'))
    os.mkdir(status_path)

  return suv_max, suv_mean, region_pd, region_data


def get_roi_metrics(imgs, pid, rois, metrics, roi_label, models, tmp_dir):
  status_path = os.path.join(tmp_dir, 'metrics_roi_done')
  targets = imgs['full']
  metric_roi = np.zeros((len(rois), len(models), len(pid), len(metrics)))

  if not os.path.exists(status_path):
    if os.path.exists(os.path.join(tmp_dir, 'metrics_roi')):
      metric_roi = joblib.load(os.path.join(tmp_dir, 'metrics_roi'))
    else:
      max_suv = [np.max(imgs['low'][i]) for i in range(len(pid))]
      for r, roi in enumerate(rois):
        for j in range(len(pid)):
          if isinstance(roi, list):
            onehot = np.isin(imgs['seg'][j], roi)
          else:
            onehot = imgs['seg'][j] == roi
          target = targets[j] * onehot / max_suv[j]
          for i, key in enumerate(models):
            img = imgs[key][j] * onehot / max_suv[j]
            m_result = get_metrics(target, img, metrics)
            for t, k in enumerate(m_result):
              metric_roi[r, i, j, t] = m_result[k]
      joblib.dump(metric_roi, os.path.join(tmp_dir, 'metrics_roi'))

    for i, arr in enumerate(metric_roi):
      save_path = os.path.join(tmp_dir, f'{roi_label[i]}_roi_metrics.csv')
      data = np.zeros((len(models)*(len(pid)+1), len(metrics)+1)).astype(str)
      for j in range(len(models)):
        item = j * (len(pid)+1)
        data[item] = models[j:j+1] + metrics
        data[item+1:item+len(pid)+1] = np.insert(arr[j].astype(str), 0, pid.reshape((1, -1)), axis=1)
      np.savetxt(save_path, data, delimiter=',', fmt="%s")
    os.mkdir(status_path)
  else:
    files = list_folder(tmp_dir, r'^.*_roi_metrics.csv')
    for i, file in enumerate(files):
      arr = np.loadtxt(os.path.join(tmp_dir, file), dtype=np.str_, delimiter=',')
      item = roi_label.index(file[:-16])
      for j in range(len(models)):
        metric_roi[item, j] = arr[j*(len(pid)+1)+1:j*(len(pid)+1)+len(pid)+1, 1:].astype(np.float64)

  return metric_roi


def draw_bar_with_err(ax, x, y, y_err, width, label):
  ax.bar(x, y, width, label=label)
  ax.errorbar(x, y, yerr=y_err, fmt='.', color='red', ecolor='black', capsize=6)


def draw_global_metrics(ax1, ax2, x, metric_arr):
  mean = np.mean(metric_arr, axis=1)
  std = np.std(metric_arr, axis=1)
  psnr_ax = ax1.twinx()

  for i, label in enumerate(labels):
    draw_bar_with_err(ax1, x[0], mean[i][0], std[i][0], width, label)
    draw_bar_with_err(psnr_ax, x[1], mean[i][1], std[i][1], width, label)
    draw_bar_with_err(ax2, x[2:], mean[i][2:], std[i][2:], width, label)
    x = x + 1 / len(metrics)

  return psnr_ax


def draw_region_metrics(ax1, ax2, x, metric_arr):
  mean = np.mean(metric_arr, axis=1)
  std = np.std(metric_arr, axis=1)
  psnr_ax = ax1.twinx()
  rela_ax = ax2.twinx()

  for i, label in enumerate(labels):
    draw_bar_with_err(ax1, x[0], mean[i][0], std[i][0], width, label)
    draw_bar_with_err(psnr_ax, x[1], mean[i][1], std[i][1], width, label)
    draw_bar_with_err(ax2, x[2], mean[i][2], std[i][2], width, label)
    draw_bar_with_err(rela_ax, x[3], mean[i][3], std[i][3], width, label)
    x = x + 1 / len(metrics)

  return psnr_ax, rela_ax


def draw_suv_dist(ax, data, label, roi: str, vmax=5):
  data = data[data['roi'] == roi]
  sns.violinplot(y='pixel', x='model', data=data, ax=ax,
                 inner='quartile')

  ax.set_title(f'the SUV Distribution of {roi}')
  ax.set_xticks(np.arange(len(label)), label)
  ax.set_ylim(0, vmax)



if __name__ == '__main__':
  root_dir = f'../'
  tmp_dir = f'./tmp'
  save_dir = get_save_dir(root_dir)
  compare_dir = []
  raw_dir = f'../raw_data'
  raw = ['low', 'full', 'seg']
  raw_name = ['30G', '240G', 'seg']
  models = ['01_unet', '07_gan', '02_va']
  models_label = ['U-Net', 'GAN', 'VA-Net']
  select_check = [0, 0, 0]

  if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)

  # data load
  for i, model in enumerate(models):
    console.supplement(f'[{i}] {model}:', level=1)
    for j, checkpoints in enumerate(save_dir[model]):
      console.supplement(f'[{j}] {checkpoints.split("/")[-2]}', level=2)
      if select_check[i] == j:
        compare_dir.append(checkpoints)
  imgs, pid = load_imgs(compare_dir, raw_dir, raw, models, raw_name, fake=True)

  roi = [5, [10, 11], [12, 13, 14], 51]
  roi_label = ['liver', 'left lung', 'right lung', 'heart']

  # Metrics
  metrics = ['SSIM', 'PSNR', 'NRMSE', 'RELA']

  metric_arr = get_metrics_table(imgs, pid, metrics,
                                 raw[:1]+models, tmp_dir, raw)
  metric_roi = get_roi_metrics(imgs, pid, roi, metrics, roi_label,
                               raw[:1]+models, tmp_dir)

  # SUV


  suv_max, suv_mean, region_pd, region_data = get_suv_data(imgs, pid, roi, roi_label,
                                                           raw[:1]+models+raw[1:2], tmp_dir)


  # Draw
  width = 0.2
  fig, axs = plt.subplots(2, 2, figsize=(12, 12))
  func = ['metric', 'roi_metric', 'suv_dis', 'suv', 'hist'][:1]

  # Metric
  if 'metric' in func:
    metric_x = np.arange(len(metrics))
    labels = ['30s Gated'] + models_label

    psnr_ax = draw_global_metrics(axs[0, 0], axs[0, 1], metric_x, metric_arr)

    axs[0, 0].set_ylim(0.92, 1)
    psnr_ax.set_ylim(35, 65)
    axs[0, 1].set_ylim(0.1, 0.8)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 0].set_xticks([0.375, 1.375], metrics[:2])
    axs[0, 1].set_xticks([2.375, 3.375], metrics[2:])
    axs[0, 0].set_title('Metrics')
    axs[0, 1].set_title('Loss')

  if 'roi_metric' in func:
    metric_x = np.arange(len(metrics))
    labels = ['30s Gated'] + models_label

    for i, roi in enumerate(roi_label[:2]):
      psnr_ax, rela_ax = draw_region_metrics(axs[i, 0], axs[i, 1],
                                             metric_x, metric_roi[i])
      axs[i, 0].set_ylim(0.996, 1)
      axs[i, 1].set_ylim(0.1, 0.5)
      psnr_ax.set_ylim(45, 85)
      axs[i, 0].legend()
      axs[i, 1].legend()
      axs[i, 0].set_xticks([0.375, 1.375], metrics[:2])
      axs[i, 1].set_xticks([2.375, 3.375], metrics[2:])
      axs[i, 0].set_title(f'Metrics of Region {roi}')
      axs[i, 1].set_title(f'Loss of Region {roi}')
      pass

  # SUV
  if 'suv_dis' in func:
    # data = region_pd[
    #   (region_pd['model'] == '05_upl') |
    #   (region_pd['model'] == 'full') |
    #   (region_pd['model'] == 'low')
    #   ]
    # sns.violinplot(y='pixel', x='roi', hue='model', data=data, ax=axs[1, 0], inner='quartile')
    types = ['30s Gated'] + models_label + ['240s Gated']

    data = region_pd[
      region_pd['pid'] == 'YHP00012016'
    ]

    draw_suv_dist(axs[0, 0], data, types, 'liver')
    draw_suv_dist(axs[0, 1], data, types, 'left lung', vmax=2)
    draw_suv_dist(axs[1, 0], data, types, 'right lung', vmax=2.5)
    draw_suv_dist(axs[1, 1], data, types, 'heart', vmax=3)
    # axs[1, 0].set_title('the SUV Distribution of ROI (U-Net)')
    # axs[1, 0].set_ylim(0, 5)

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
                        f'(NRMSE = {metric_arr[0][0][2]:.2f})')
    hist_joint(fig, axs[0, 1], imgs['full'][0], imgs['05_upl'][0],
               '240s Gated', 'U-Net', -3, 3)
    axs[0, 1].set_title(f'Joint Voxel Histogram - Log Scale '
                        f'(NRMSE = {metric_arr[1][0][2]:.2f})')
    hist_joint(fig, axs[1, 0], imgs['full'][0], imgs['03_uadap'][0],
               '240s Gated', 'VA-Net 1', -3, 3)
    axs[1, 0].set_title(f'Joint Voxel Histogram - Log Scale '
                        f'(NRMSE = {metric_arr[2][0][2]:.2f})')
    hist_joint(fig, axs[1, 1], imgs['full'][0], imgs['06_puadap'][0],
               '240s Gated', 'VA-Net 2', -3, 3)
    axs[1, 1].set_title(f'Joint Voxel Histogram - Log Scale '
                        f'(NRMSE = {metric_arr[2][0][3]:.2f})')
  plt.show()


  pass

