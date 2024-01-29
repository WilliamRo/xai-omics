import os

import numpy as np
import matplotlib.pyplot as plt

from roma import console


regions = np.array(['others', 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver',
                    'stomach', 'pancreas', 'adrenal_gland_right', 'adrenal_gland_left',
                    'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                    'lung_middle_lobe_right', 'lung_lower_lobe_right', 'esophagus',
                    'trachea', 'thyroid_gland', 'small_bowel', 'duodenum', 'colon',
                    'urinary_bladder', 'prostate', 'kidney_cyst_left', 'kidney_cyst_right',
                    'sacrum', 'vertebrae_S1', 'vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3',
                    'vertebrae_L2', 'vertebrae_L1', 'vertebrae_T12', 'vertebrae_T11',
                    'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8', 'vertebrae_T7',
                    'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
                    'vertebrae_T2', 'vertebrae_T1', 'vertebrae_C7', 'vertebrae_C6',
                    'vertebrae_C5', 'vertebrae_C4', 'vertebrae_C3', 'vertebrae_C2',
                    'vertebrae_C1', 'heart', 'aorta', 'pulmonary_vein', 'brachiocephalic_trunk',
                    'subclavian_artery_right', 'subclavian_artery_left',
                    'common_carotid_artery_right', 'common_carotid_artery_left',
                    'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
                    'atrial_appendage_left', 'superior_vena_cava', 'inferior_vena_cava',
                    'portal_vein_and_splenic_vein', 'iliac_artery_left', 'iliac_artery_right',
                    'iliac_vena_left', 'iliac_vena_right', 'humerus_left', 'humerus_right',
                    'scapula_left', 'scapula_right', 'clavicula_left', 'clavicula_right',
                    'femur_left', 'femur_right', 'hip_left', 'hip_right', 'spinal_cord',
                    'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left',
                    'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right',
                    'autochthon_left', 'autochthon_right', 'iliopsoas_left', 'iliopsoas_right',
                    'brain', 'skull', 'rib_right_4', 'rib_right_3', 'rib_left_1', 'rib_left_2',
                    'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
                    'rib_left_8', 'rib_left_9', 'rib_left_10', 'rib_left_11', 'rib_left_12',
                    'rib_right_1', 'rib_right_2', 'rib_right_5', 'rib_right_6', 'rib_right_7',
                    'rib_right_8', 'rib_right_9', 'rib_right_10', 'rib_right_11',
                    'rib_right_12', 'sternum', 'costal_cartilages'])


def calc_region(img, seg, region):
  length = len(regions)
  suv_max = np.zeros(length, dtype=np.float64)
  suv_mean = np.zeros(length, dtype=np.float64)
  for item in region:
    img_region = img*(seg == item)
    suv_max[item] = np.max(img_region)
    suv_mean[item] = np.mean(img_region)
  return suv_max, suv_mean


def calc_suv_statistic(imgs, segs, region):
  suv_maxs = []
  suv_means = []
  length = len(imgs)
  for i in range(length):
    suv_max, suv_mean = calc_region(imgs[i], segs[i], region)
    suv_maxs.append(suv_max)
    suv_means.append(suv_mean)
    console.print_progress(i, length)
  console.clear_line()
  return suv_maxs, suv_means


def load_suv_stat(imgs, segs, path, pid, name, region=None):
  region = list(range(len(regions))) if region is None else region
  path_max = os.path.join(path, f'{name}_suv_max.csv')
  path_mean = os.path.join(path, f'{name}_suv_mean.csv')
  if not os.path.exists(path_max):
    suv_maxs, suv_means = calc_suv_statistic(imgs, segs, region)
    header_row = np.insert(regions, 0, "")
    header_col = pid
    save_max = add_title_row_col(suv_maxs, header_row, header_col)
    save_mean = add_title_row_col(suv_means, header_row, header_col)
    np.savetxt(path_max, save_max, delimiter=',', fmt="%s")
    np.savetxt(path_mean, save_mean, delimiter=',', fmt="%s")
  else:
    suv_maxs = np.loadtxt(path_max, delimiter=',', dtype=np.str_)
    suv_means = np.loadtxt(path_mean, delimiter=',', dtype=np.str_)
    suv_maxs = suv_maxs[1:, 1:].astype(np.float64)
    suv_means = suv_means[1:, 1:].astype(np.float64)
  suv_max = np.mean(suv_maxs, axis=0), np.std(suv_maxs, axis=0)
  suv_mean = np.mean(suv_means, axis=0), np.std(suv_means, axis=0)
  return suv_max, suv_mean


def add_title_row_col(data, header_row, header_col):
    return np.vstack((header_row, np.hstack((header_col.reshape(-1, 1), data))))


def draw_one_bar(ax, x, data, width, roi, label):
  ax.bar(x, data[0][roi], width, label=label)
  ax.errorbar(x, data[0][roi], yerr=data[1][roi], fmt='.',
              color='red', ecolor='black', capsize=6)


def set_ax(axs, titles, x, roi, legend=True):
  xlabel = regions[roi]
  xlabel = ['\n'.join(i.split('_')) for i in xlabel]
  for ax, title in zip(axs, titles):
    ax.set_xticks(x, xlabel, rotation=0)
    ax.set_xlabel('Regions')
    ax.set_ylabel('$SUV$')
    ax.set_title(title)
    if legend:
      ax.legend()


def get_mean_std_metric(truths, data, metrics):
  from xomics.data_io.utils.metrics_calc import get_metrics
  metric = np.zeros((len(truths), len(metrics)))
  for progress, i in enumerate(range(len(truths))):
    console.print_progress(progress, len(truths))
    arr1 = truths[i]
    arr2 = data[i] / np.max(arr1)
    arr1 = arr1 / np.max(arr1)
    s_metrics = get_metrics(arr1, arr2, metrics, data_range=1.0)
    for j, key in enumerate(metrics):
      metric[i, j] = s_metrics[key]
  return metric


def load_metric_stat(truths, data, metrics, path, pid, name):
  metric_path = os.path.join(path, f'{name}_metrics.csv')
  if not os.path.exists(metric_path):
    metric = get_mean_std_metric(truths, data, metrics)
    save_metric = add_title_row_col(metric, [""]+metrics, pid)
    np.savetxt(metric_path, save_metric, delimiter=',', fmt="%s")
  else:
    metric = np.loadtxt(metric_path, delimiter=',', dtype=np.str_)
    metric = metric[1:, 1:].astype(np.float64)
  metric_mean, metric_std = np.mean(metric, axis=0), np.std(metric, axis=0)
  console.clear_line()
  return metric_mean, metric_std


def metric_text(ax, ax_psnr, metric, x, width):
  for i, value in enumerate(metric[0][:-1]):
    if i == 0:
      value *= 0.5
    elif i == 1:
      value *= 1.6
    else:
      value *= 1.2
    ax.annotate(f'{value:.2f}\n±{metric[1][i]:.2f}',
                       (x[i] - width / 2, value),
                       ha='center', va='bottom')
  ax_psnr.annotate(f'{metric[0][-1]:.2f}\n±{metric[1][-1]:.2f}',
                   (x[-1] - width / 2, metric[0][-1] * 0.5),
                   ha='center', va='bottom')


def hist_joint(fig, ax, img1, img2, xlable, ylabel, min_val, max_val, bins=600):
  img1, img2 = np.vstack(img1), np.vstack(img2)
  img1, img2 = img1.flatten(), img2.flatten()
  epsilon = 1e-10
  img1[img1 <= 0], img2[img2 <= 0] = epsilon, epsilon

  # print(img1.shape, img2.shape)
  img1, img2 = np.log(img1), np.log(img2)
  hist2D, x_edges, y_edges = np.histogram2d(img1, img2, bins=bins,
                                            range=[[min_val, max_val],
                                                   [min_val, max_val]])
  # print(x_edges, y_edges)
  hist2D = hist2D.T.astype(np.int64)
  nonzero_coords = np.transpose(np.nonzero(hist2D))
  pos = np.repeat(nonzero_coords,
                  hist2D[nonzero_coords[:, 0], nonzero_coords[:, 1]], axis=0)
  x = x_edges[pos[:, 0]]
  y = y_edges[pos[:, 1]]
  coefficients = np.polyfit(x, y, 1)
  k = coefficients[0]
  b = coefficients[1]
  x1 = np.arange(-3, 3)
  y1 = k * x1 + b

  p = np.poly1d(coefficients)
  y_pred = p(x)  # 预测值
  residuals = y - y_pred  # 残差
  ssr = np.sum(residuals ** 2)  # 残差平方和
  sst = np.sum((y - np.mean(y)) ** 2)  # 总平方和
  r_squared = 1 - (ssr / sst)  # R方

  im = ax.imshow(hist2D, origin='lower', cmap='jet',
                 extent=[min_val, max_val, min_val, max_val])

  ax.plot(x1, y1, color='white', linestyle='--')
  equation = f'$y = {k:.2f}x{b:+.2f}, R^2 = {r_squared:.2f}$'
  ax.text(-2, 2, equation, color='white')

  ax.set_xticks(np.arange(min_val, max_val + 1))
  ax.set_yticks(np.arange(min_val, max_val + 1))
  ax.set_xlabel(xlable)
  ax.set_ylabel(ylabel)
  ax.set_title('Joint Voxel Histogram - Log Scale')
  fig.colorbar(im)

  return coefficients


def violin_plot_roi(ax, data, seg, roi, percent=99.9):
  data_list = []
  data = np.vstack(data)
  for rid in roi:
    img = data*(np.vstack(seg) == rid)
    img = img.flatten()
    img = img[img > 0]
    p1 = np.percentile(img, percent)
    img = img[img < p1]
    data_list.append(img)
  ax.violinplot(dataset=data_list, showmedians=True)


def violin_plot(ax, data, seg, roi, xlable, percent=99.9):
  data_list = []
  sl = np.vstack(seg) == roi
  for img in data:
    img = np.vstack(img)
    img = img * sl
    img = img.flatten()
    img = img[img > 0]
    p1 = np.percentile(img, percent)
    img = img[img < p1]
    data_list.append(img)
  ax.violinplot(dataset=data_list, showmedians=True)
  ax.set_xticks(np.arange(1, len(data)+1), xlable)
  ax.set_ylabel('SUV')
  ax.set_title(f'The SUV of {regions[roi]}')

