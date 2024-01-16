import numpy as np
import matplotlib.pyplot as plt

from utils.statistics import regions, load_suv_stat, \
  load_metric_stat
from xomics.objects.general_mi import GeneralMI








if __name__ == '__main__':
  img_dict = {}
  data = np.genfromtxt(r'../../data/02-RLD/rld_data.csv', delimiter=',', dtype=str)
  types = data[0][1:]
  pid = data[1:, 0]
  path_array = data[1:, 1:]

  for i, type_name in enumerate(types):
    img_path = path_array[:, i]
    img_dict[type_name] = {'path': img_path}

  img_type = {
    'CT': ['CT'],
    'PET': ['30G', '240G'],
    'MASK': ['CT_seg'],
    'STD': ['30G']
  }

  test = GeneralMI(img_dict, ['30G', 'CT_seg'], ['240G'], pid, img_type=img_type)
  # test.process_param['norm'] = 'PET'
  test.process_param['shape'] = [440, 440, 256]
  # test.process_param['percent'] = 99.9
  # test.process_param['ct_window'] = [50, 500]

  region_nums = list(range(len(regions)))

  width = 0.3

  roi = [5, 10, 11, 12, 13, 14, 51]
  path = './'

  suv_max_30, suv_mean_30 = load_suv_stat(test.images['30G'], test.images['CT_seg'],
                                          path, test.pid, '30sG')
  suv_max_240, suv_mean_240 = load_suv_stat(test.labels['240G'], test.images['CT_seg'],
                                            path, test.pid, '240sG')

  x = np.arange(len(roi))

  fig, axs = plt.subplots(2, 2, figsize=(32, 12))
  fig.subplots_adjust(hspace=0.5, wspace=0.5)

  # draw_one_bar(axs[1][0], x, suv_max_30, width, roi, '30s Gated')
  # draw_one_bar(axs[1][0], x+width, suv_max_240, width, roi, '240s Gated')
  #
  # draw_one_bar(axs[1][1], x, suv_mean_30, width, roi, '30s Gated')
  # draw_one_bar(axs[1][1], x+width, suv_mean_240, width, roi, '240s Gated')
  # set_ax([axs[1][0], axs[1][1]], ['$SUV_{max}$', '$SUV_{mean}$'], x, roi)
  #
  metrics = ['SSIM', 'NRMSE', 'RELA', 'PSNR']
  input_metric = load_metric_stat(test.labels['240G'][:2], test.images['30G'][:2],
                                  metrics, path, test.pid[:2], '30sG')
  metric_x = np.arange(len(metrics))
  axs[0][0].bar(metric_x[:-1] - width / 2, input_metric[0][:-1], label='30s Gated')
  axs[0][0].errorbar(metric_x[:-1] - width / 2, input_metric[0][:-1], yerr=input_metric[1][:-1],
                     fmt='.', color='red', ecolor='black', capsize=6)
  axs[0][0].set_xticks(metric_x, metrics)
  ax_psnr = axs[0][0].twinx()
  ax_psnr.bar(metric_x[-1] - width / 2, input_metric[0][-1], label='30s Gated')
  ax_psnr.errorbar(metric_x[-1] - width / 2, input_metric[0][-1], yerr=input_metric[1][-1],
                   fmt='.', color='red', ecolor='black', capsize=6)

  for i, value in enumerate(input_metric[0][:-1]):
    if i == 0:
      value *= 0.5
    elif i == 1:
      value *= 1.6
    else:
      value *= 1.2
    axs[0][0].annotate(f'{value:.2f}\n±{input_metric[1][i]:.2f}',
                       (metric_x[i] - width / 2, value),
                       ha='center', va='bottom')
  ax_psnr.annotate(f'{input_metric[0][-1]:.2f}\n±{input_metric[1][-1]:.2f}',
                   (metric_x[-1] - width / 2, input_metric[0][-1]*0.5),
                   ha='center', va='bottom')

  # hist_joint(fig, axs[0, 1], test.images['30G'][:3], test.labels['240G'][:3],
  #            '30s Gated', '240s Gated', -3, 3)
  # fig.colorbar(axs[0, 1])

  # violin_plot_roi(axs[0, 1], test.images['30G'][:2], test.images['CT_seg'][:2], roi)
  # violin_plot(axs[1, 1], [test.images['30G'][:2], test.labels['240G'][:2]],
  #             test.images['CT_seg'][:2], 5, ['30s Gated', '240s Gated'])
  plt.show()


