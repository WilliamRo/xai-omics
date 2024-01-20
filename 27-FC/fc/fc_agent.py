import os
import numpy as np
import openpyxl
import pickle

from fc.fc_set import FCSet
from collections import OrderedDict
from tframe.data.base_classes import DataAgent
from roma import console
from copy import deepcopy


class FCAgent(DataAgent):

  TFD_FILE_NAME = 'fc.tfd'

  @classmethod
  def load(cls):

    from fc_core import th
    fc_set = cls.load_as_tframe_data(th.data_dir)

    if th.label_type == 'trg_01_23':
      targets = fc_set.data_dict['label_trg_01_23']
    elif th.label_type == 'trg_012_3':
      targets = fc_set.data_dict['label_trg_012_3']
    elif th.label_type == 'trg_01_2_3':
      targets = fc_set.data_dict['label_trg_01_2_3']
    else:
      assert TypeError('Wrong Label Type!!!')

    features = fc_set.features
    pids = fc_set.data_dict['pids']

    ratio = th.ratio_of_dataset.split(':')
    ratio = [int(a) for a in ratio]

    data_dict = {'features': features, 'targets': targets, 'pids': pids}
    fc_set = FCSet(
      data_dict=data_dict, NUM_CLASSES=th.num_classes,
      classes=['A', 'B', 'C'][:th.num_classes])

    if th.cross_validation:
      size = [fc_set.size // th.k_fold for _ in range(th.k_fold - 1)]
      size = size + [fc_set.size - sum(size)]

      folds = fc_set.split(
        size, names=[f'fold_{i + 1}' for i in range(th.k_fold)],
        over_classes=True)

      train_val_set = combine_dataset(
        [folds[i] for i in range(th.k_fold) if i != th.val_fold_index % th.k_fold])
      train_set, valid_set = train_val_set.split(
        9, 1, names=['TrainSet', 'ValidSet'], over_classes=True )

      test_set = folds[th.val_fold_index % th.k_fold]
      test_set.name = 'TestSet'
    else:
      train_set, valid_set, test_set = fc_set.split(
        ratio[0] * 0.8, ratio[0] * 0.2, ratio[1],
        names=['TrainSet', 'ValidSet', 'TestSet'], over_classes=True)

    # feature_mean = np.mean(train_set.features, axis=0)
    # feature_std = np.std(train_set.features, axis=0)
    # train_set.features = (train_set.features - feature_mean) / feature_std
    # valid_set.features = (valid_set.features - feature_mean) / feature_std
    # test_set.features = (test_set.features - feature_mean) / feature_std

    return train_set, valid_set, test_set


  @classmethod
  def load_as_tframe_data(cls, data_dir, *args, **kwargs) -> FCSet:
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return FCSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw datas
    console.show_status('Trying to convert raw datas to tframe DataSet ...')
    image_dict = cls.load_as_numpy_arrays(data_dir)

    data_set = FCSet(data_dict=image_dict, name='FCSet')

    # Show status
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving datas set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    # Wrap and return

    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir) -> OrderedDict:

    print(data_dir)
    image_dict = OrderedDict()
    selected = True

    if selected:
      features_path = os.path.join(data_dir, 'selected_feature_dict.pkl')
    else:
      features_path = os.path.join(
        os.path.dirname(os.path.dirname(data_dir)),
        'data/02-PET-CT-Y1/features/feature_dict.pkl')

    label_path = os.path.join(
      os.path.dirname(os.path.dirname(data_dir)),
      'data/02-PET-CT-Y1/sg_trg.xlsx')

    # Targets
    workbook = openpyxl.load_workbook(label_path)
    sheet = workbook['Sheet1']
    col_trg23, col_trg3 = sheet['D'], sheet['E']

    label_trg01_23 = np.array(
      [c.value for c in col_trg23 if c.value != 'trg23'])
    label_trg012_3 = np.array([c.value for c in col_trg3 if c.value != 'trg3'])
    assert len(label_trg01_23) == len(label_trg012_3) == 108
    label_trg01_2_3 = np.array([
      t1 + t2 for t1, t2 in zip(label_trg012_3, label_trg01_23)])

    label_trg01_23 = np.eye(2)[label_trg01_23.flatten().astype(int)].astype(np.uint8)
    label_trg012_3 = np.eye(2)[label_trg012_3.flatten().astype(int)].astype(np.uint8)
    label_trg01_2_3 = np.eye(3)[label_trg01_2_3.flatten().astype(int)].astype(np.uint8)

    # Features
    if selected:
      with open(features_path, 'rb') as pickle_file:
        feature_dict = pickle.load(pickle_file)
        feature_name = feature_dict['feature_name']
        features = np.array(feature_dict['x_lasso1'])
        label_trg01_23 = np.array(feature_dict['y_lasso1'])
        label_trg01_23 = np.eye(2)[label_trg01_23.flatten().astype(int)].astype(np.uint8)
    else:
      with open(features_path, 'rb') as pickle_file:
        feature_dict = pickle.load(pickle_file)

      features = []
      for key in feature_dict:
        if key == 'feature_name': feature_name = feature_dict[key]
        else: features.append(feature_dict[key])
      features = np.array(features)

    image_dict['features'] = features
    image_dict['label_trg_01_23'] = label_trg01_23
    image_dict['label_trg_012_3'] = label_trg012_3
    image_dict['label_trg_01_2_3'] = label_trg01_2_3
    image_dict['feature_name'] = feature_name

    return image_dict


def ratio_to_realnum(ratio: list, total_num: int):
  assert len(ratio) > 1
  parts = [int((r / sum(ratio)) * total_num) for r in ratio[:-1]]
  parts.append(total_num - sum(parts))
  assert sum(parts) == total_num

  return parts


def sorting_key(item):
  return int(item.split('_')[0])


def combine_dataset(input):
  items = list(input[0].data_dict.keys())
  data_dict = {}
  for i in items:
    data_dict[i] = np.concatenate([s.data_dict[i] for s in input], axis=0)

  return FCSet(
    data_dict=data_dict, NUM_CLASSES=input[0].num_classes,
    classes=input[0].properties['classes'])



if __name__ == '__main__':
  agent = FCAgent()
  train_set, val_set, test_set = agent.load()
  # pids = ['0002_huangchengxia', '0003_shaoqiquan', '0004_yuyongming', '0005_wuhanqing', '0006_yaoguolin', '0008_jiepeiyi', '0009_shenturuhong', '0010_wuwentou', '0011_yanglixin', '0013_zhoubaorong', '0014_hebaoshan', '0015_dingjinrui', '0016_fangchengde', '0017_wangshuigen', '0018_bishilun', '0019_chenmusong', '0020_lihongchun', '0021_lukaoren', '0022_huyunda', '0023_zhangbairong', '0025_chenhui', '0026_nixuzhong', '0027_xieenfu', '0028_gaowenhua', '0029_xumingshou', '0030_heouliang', '0031_liuruyong', '0032_yanxinkao', '0033_chusongnan', '0034_wudixun', '0035_jinjianqiang', '0036_zhuzhangfu', '0037_yuanyanlei', '0038_huangfujinlian', '0039_xiaominsu', '0040_caixingsheng', '0041_wuxinfu', '0042_qianrongyou', '0043_xuyuying', '0044_yaojinfa', '0045_pengyonglu', '0046_niweiliang', '0047_tangxin', '0048_yupeizhu', '0049_wangrilong', '0050_jiangruijuan', '0051_luwenwei', '0052_xiejian', '0053_zhangcaifa', '0054_xuyine', '0055_meichangrong', '0056_zhangchengkun', '0057_shenasan', '0058_fengjianming', '0059_yejiayuan', '0060_yeyizhen', '0061_qianlijun', '0062_dingyonghong', '0063_zhouzumao', '0064_yuxianping', '0065_wangnanxian', '0066_yanguomin', '0067_chenshaozeng', '0068_fanyongai', '0069_chenmingrong', '0070_lirong', '0071_zhengjiuxing', '0072_wengjuliang', '0073_yelibing', '0074_lulianrong', '0076_zhangsuming', '0077_zhaozuofang', '0078_chenjianguo', '0079_tangpeiyan', '0080_humeiling', '0081_zhangkerong', '0082_lukuanyin', '0083_wangyongxiao', '0084_hushihe', '0085_cenghaitang', '0086_shaozengbao', '0087_chenyongpei', '0088_luchaohua', '0091_xushuiyun', '0092_guotuxian', '0093_zhaodongjie', '0094_caozengkun', '0095_fangjingming', '0096_xuguoquan', '0097_huangminghao', '0098_xuyoufeng', '0099_lichaocong', '0100_qianshuping', '0101_yeliangfa', '0102_lixuean', '0103_zhangxiuling', '0104_zhouwenping', '0105_liyuerong', '0106_huangruchang', '0107_zhongshanqing', '0108_shenliansong', '0109_yangbenbi', '0110_yaopengfei', '0111_mameixian', '0112_qianashui', '0113_guguangyue', '0114_zhengbinghui', '0115_wangqingzhu']
  # from fc_core import th
  # fc_set = agent.load_as_tframe_data(th.data_dir)
  #
  # data_dict = fc_set.data_dict
  # data_dict['pids'] = np.array(pids, dtype=object)
  #
  # data_set = FCSet(data_dict=data_dict, name='FCSet')
  #
  #
  # file_path = os.path.join(th.data_dir, agent.TFD_FILE_NAME)
  # console.show_status('Saving datas set ...')
  # data_set.save(file_path)
  # console.show_status('Data set saved to {}'.format(file_path))

  print()





