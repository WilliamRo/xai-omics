import numpy as np
import matplotlib.pyplot as plt

from roma import console


regions = np.array(['background', 'spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver',
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


def calc_suv_statistic(imgs, segs, region, percent=99.9):
  suv_maxs = []
  suv_means = []
  length = len(imgs)
  for i in range(length):
    suv_max, suv_mean = calc_region(imgs[i], segs[i], region)
    suv_maxs.append(suv_max)
    suv_means.append(suv_mean)
    console.print_progress(i, length)
  console.clear_line()
  if percent is not None:
    pass
  suv_max = np.mean(suv_maxs, axis=0), np.std(suv_maxs, axis=0)
  suv_mean = np.mean(suv_means, axis=0), np.std(suv_means, axis=0)
  return suv_max, suv_mean


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
  for i in range(len(truths)):
    arr1 = truths[i]
    arr2 = data[i] / np.max(arr1)
    arr1 = arr1 / np.max(arr1)
    s_metrics = get_metrics(arr1, arr2, metrics, data_range=1.0)
    for j, key in enumerate(metrics):
      metric[i, j] = s_metrics[key]
  metric_mean, metric_std = np.mean(metric, axis=0), np.std(metric, axis=0)
  return metric_mean, metric_std


def hist_joint(fig, ax, img1, img2, xlable, ylabel, min_val, max_val):
  img1, img2 = np.vstack(img1), np.vstack(img2)
  img1, img2 = img1.flatten(), img2.flatten()
  epsilon = 1e-10
  img1[img1 <= 0], img2[img2 <= 0] = epsilon, epsilon
  # print(img1.shape, img2.shape)
  img1, img2 = np.log(img1), np.log(img2)
  hist2D, x_edges, y_edges = np.histogram2d(img1, img2, bins=600,
                                            range=[[min_val, max_val],
                                                   [min_val, max_val]])
  # print(x_edges, y_edges)
  im = ax.imshow(hist2D.T, origin='lower', cmap='jet',
                 extent=[min_val, max_val, min_val, max_val])
  ax.set_xticks(np.arange(min_val, max_val + 1))
  ax.set_yticks(np.arange(min_val, max_val + 1))
  ax.set_xlabel(xlable)
  ax.set_ylabel(ylabel)
  ax.set_title('Joint Voxel Histogram - Log Scale')
  fig.colorbar(im)


def violin_plot_roi(ax, data, seg, roi, percent=99.9):
  data_list = []

  for rid in roi:
    tmp = []
    for i in range(len(data)):
      tmp.append(data[i]*(seg[i] == rid))
    img = np.vstack(tmp)
    img = img.flatten()
    img = img[img > 0]
    p1 = np.percentile(img, percent)
    img = img[img < p1]
    data_list.append(img)
  ax.violinplot(dataset=data_list, showmedians=True)



def violin_plot(ax, data, seg, roi, xlable, percent=99.9):
  data_list = []
  for img in data:
    tmp = []
    for i in range(len(seg)):
      tmp.append(img[i] * (seg[i] == roi))
    img = np.vstack(tmp)
    img = img.flatten()
    img = img[img > 0]
    p1 = np.percentile(img, percent)
    img = img[img < p1]
    data_list.append(img)
  ax.violinplot(dataset=data_list, showmedians=True)
  ax.set_xticks(np.arange(1, len(data)+1), xlable)
  ax.set_title(f'The SUV of {regions[roi]}')

