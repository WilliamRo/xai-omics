import copy
import numpy as np
import SimpleITK as sitk
import joblib

from roma import console
from xomics import MedicalImage
from xomics.data_io.utils.preprocess import get_suv_factor
from xomics.data_io.utils.raw_rw import reshape_image, resize_image_itk, \
  resample_image_by_spacing


#  images_dict =
#  {'type1':
#    {'path': [path], 'itk_img': [image_itk], 'img':[img_itk]}
#  }


class Indexer:
  """
  An internal Class to implement some functions
  """
  def __init__(self, obj, name='noname', type_name=None, key_name=None):
    """
    :param obj: GeneralMI object
    :param name: name of the data, [images, labels]
    :param type_name: type of the data, ['30G', '240S', 'CT', 'CT_seg', ...]
    :param key_name: key of the data, [img, img_itk]
    """
    self.mi: GeneralMI = obj
    assert name in ['images', 'labels', 'noname']
    self.name = name
    self.type_name = type_name
    self._data = self.mi.images_dict[type_name] if type_name else None
    self.key_name = key_name

  @property
  def data(self):
    return self._data

  @data.getter
  def data(self):
    self._data = self.mi.images_dict[self.type_name]
    return self._data

  def __iter__(self):
    self.index = 0
    return self

  def __next__(self):
    if self.index >= len(self):
      raise StopIteration
    value = self[self.index]
    self.index += 1
    return value

  def __getitem__(self, item):
    if isinstance(item, (int, np.int_)):
      item = int(item)
      return self.get_data(item)
    elif isinstance(item, (list, np.ndarray)):
      data = []
      for num, i in enumerate(item):
        console.print_progress(num, len(item))
        data.append(self.get_data(i))
      console.clear_line()
      return data
    elif isinstance(item, slice):
      start = item.start if item.start else 0
      stop = item.stop if item.stop else len(self)
      step = item.step if item.step else 1
      if start > stop:
        start, stop = stop+1, start+1
      if step < 0:
        iterator = iter(range(start, stop, -step))
      else:
        iterator = iter(range(start, stop, step))
      data = []
      for num, i in enumerate(iterator):
        console.print_progress(num, int((stop-start)/step))
        data.append(self.get_data(i))
      console.clear_line()
      if step < 0:
        data.reverse()
      return data
    elif isinstance(item, str):
      return self._data[item]
    else:
      raise TypeError('Invalid index type')

  def __len__(self):
    return len(self._data['path'])

  def get_data(self, item):
    pass


class ItkIndexer(Indexer):
  """
  get simpleITK image object from raw image file
  """
  def __init__(self, process_func=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.PROCESS_FUNC = process_func

  def get_data(self, item):
    if self.data[self.key_name][item] is None:
      assert self.data['path'][item] != ''
      self.data[self.key_name][item] = GeneralMI.load_img(self.data['path'][item])
      if self.PROCESS_FUNC is not None:
        self.data[self.key_name][item] = self.PROCESS_FUNC(self.data[
                                                             self.key_name
                                                           ][item],
                                                           self.data,
                                                           self.type_name,
                                                           self.name, item,
                                                           self.type)
    tmp = self.data[self.key_name][item]
    if self.mi.LOW_MEM:
      self.data[self.key_name][item] = None
    return tmp

  @property
  def type(self):
    for key, item in self.mi.img_type.items():
      if self.type_name in item:
        return key
    return 'Unknown'


class ImgIndexer(ItkIndexer):
  """
  get processed simpleitk obj from raw simpleitk obj
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_data(self, item):
    if self.data[self.key_name][item] is None:
      if self.name == 'images':
        img = self.mi.images_raw[self.type_name].itk[item]
      elif self.name == 'labels':
        img = self.mi.labels_raw[self.type_name].itk[item]
      else:
        img = None
      if self.PROCESS_FUNC is None:
        self.data[self.key_name][item] = img
      else:
        self.data[self.key_name][item] = self.PROCESS_FUNC(img, self.data,
                                                           self.type_name,
                                                           self.name,
                                                           item, self.type)
    tmp = self.data[self.key_name][item]
    if self.mi.LOW_MEM:
      self.data[self.key_name][item] = None
    return sitk.GetArrayFromImage(tmp)

  @property
  def itk(self):
    return ItkIndexer(self.mi.raw_process, self.mi, name=self.name,
                      type_name=self.type_name, key_name='img_itk')


class Dicter(ImgIndexer):
  """
  make data like a dictionary
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __getitem__(self, item):
    assert isinstance(item, (str, np.str_, int))
    if isinstance(item, int):
      if self.name == 'images':
        assert item < len(self.mi.image_keys)
        return ImgIndexer(self.PROCESS_FUNC, self.mi, key_name=self.key_name,
                          name=self.name, type_name=self.mi.image_keys[item])
      elif self.name == 'labels':
        assert item < len(self.mi.label_keys)
        return ImgIndexer(self.PROCESS_FUNC, self.mi, key_name=self.key_name,
                          name=self.name, type_name=self.mi.label_keys[item])

    item = str(item)
    if self.name == 'images':
      assert item in self.mi.image_keys
    elif self.name == 'labels':
      assert item in self.mi.label_keys
    return ImgIndexer(self.PROCESS_FUNC, self.mi, key_name=self.key_name,
                      name=self.name, type_name=item)

  # region: medical_image compatible

  def __len__(self):
    if self.name == 'images':
      return len(self.mi.image_keys)
    elif self.name == 'labels':
      return len(self.mi.label_keys)

  def keys(self):
    if self.name == 'images':
      return self.mi.image_keys
    elif self.name == 'labels':
      return self.mi.label_keys

  # endregion: medical_image compatible



class GeneralMI:
  """
  a general data framework for pet/ct
  """
  def __init__(self, images_dict, image_keys=None, label_keys=None,
               pid=None, process_param=None, img_type=None):
    self.IMG_TYPE = 'nii.gz'
    self.PRO_TYPE = 'pkl'
    self.LOW_MEM = False
    self.pid = pid

    self._image_keys, self._label_keys = [], []
    self.images_dict = images_dict
    self.custom_dict = {}
    self.image_keys = image_keys
    self.label_keys = label_keys

    self.raw_process = self.pre_process
    self.data_process = self.process
    self.img_type = img_type
    self.process_param = {
      'ct_window': None,
      'norm': None,  # only min-max,
      'shape': None,  # [320, 320, 240]
      'crop': None,  # [30, 30, 10]
      'clip': None,  # [1, None]
      'percent': None,  # 99.9
    } if process_param is None else process_param

  # region: class relevant

  def __len__(self):
    return len(self.pid)

  def __getitem__(self, item):
    assert item is not np.int_
    pid = self.pid[item]
    image_dict = copy.deepcopy(self.images_dict)
    for key in image_dict.keys():
      image_dict[key]['path'] = self.images_dict[key]['path'][item]
      if key in self.image_keys or key in self.label_keys:
        image_dict[key]['img_itk'] = self.images_dict[key]['img_itk'][item]

    return GeneralMI(image_dict, image_keys=self.image_keys,
                     label_keys=self.label_keys,
                     pid=pid, process_param=self.process_param,
                     img_type=self.img_type)

  def index(self, pid):
    index = np.where(self.pid == pid)
    if len(index[0]) == 0:
      raise ValueError('pid not found')
    return int(index[0][0])

  def _remove(self, pid):
    for k, v in self.images_dict.items():
      self.images_dict[k]['path'] = np.delete(v['path'], pid)
    self.pid = np.delete(self.pid, pid)

  def remove(self, pid):
    if isinstance(pid, (str, np.str_)):
      self.remove(self.index(pid))
    elif isinstance(pid, (int, np.int_)):
      self._remove(pid)
    elif isinstance(pid, (list, np.ndarray)):
      assert len(pid) != 0
      in_type = set([type(_) for _ in pid])
      assert len(in_type) == 1

      if in_type == {int}:
        self._remove(pid)
        return
      ids = []
      for p in pid:
        ids.append(self.index(p))
      self._remove(ids)

  def rm_void_data(self):
    use_keys = set(self.image_keys + self.label_keys)
    rm_pids = []
    for i, pi in enumerate(self.pid):
      for key in use_keys:
        if self.images_dict[key]['path'][i] == '':
          rm_pids.append(pi)
          break
    self.remove(rm_pids)

  def clean_mem(self):
    for key in self.images_dict.keys():
      length = len(self.images_dict[key]['path'])
      self.images_dict[key]['img_itk'] = np.array([None] * length)
      self.images_dict[key]['img'] = np.array([None] * length)

  # endregion: class relevant

  # region: data process

  def pre_process(self, img, data, data_type, name, item, img_type):
    new_img = img

    if img_type == 'MASK':
      new_img = resample_image_by_spacing(new_img, (1.0, 1.0, 1.0), sitk.sitkNearestNeighbor)
    else:
      new_img = resample_image_by_spacing(new_img, (1.0, 1.0, 1.0))

    if img_type == 'PET':
      new_img = self.suv_transform(new_img, self.get_tags(item))
      if self.process_param.get('percent'):
        new_img = self.percentile(new_img, self.process_param['percent'])
    elif img_type == 'CT':
      if self.process_param.get('ct_window'):
        wc = self.process_param['ct_window'][0]
        wl = self.process_param['ct_window'][1]
        new_img = sitk.IntensityWindowing(new_img, wc - wl/2, wc + wl/2, 0, 255)
      else:
        new_img = sitk.RescaleIntensity(new_img, 0, 255)
    if img_type != 'PET':
      std_type = self.STD_key
      if std_type in self.image_keys:
        new_img = resize_image_itk(new_img, self.images[std_type].itk[0])
      elif std_type in self.label_keys:
        new_img = resize_image_itk(new_img, self.labels[std_type].itk[0])
    if self.process_param.get('crop'):
      new_img = GeneralMI.crop_by_margin(new_img, self.process_param['crop'])
    if self.process_param.get('shape'):
      new_img = reshape_image(new_img, self.process_param['shape'])

    return new_img

  def process(self, img, data, data_type, name, item, img_type):
    new_img: sitk.Image = img
    if 'MASK' == img_type:
      pass
    elif 'CT' == img_type:
      if self.process_param.get('norm'):
        new_img = sitk.RescaleIntensity(new_img, 0.0, 1.0)
    elif 'PET' == img_type:
      if self.process_param.get('clip'):
        clip = self.process_param['clip']
        if clip[0] is None:
          clip[0] = np.min(sitk.GetArrayFromImage(new_img))
        elif clip[1] is None:
          clip[1] = np.max(sitk.GetArrayFromImage(new_img))
        new_img = sitk.IntensityWindowing(new_img)
      if self.process_param.get('norm'):
        if name == 'images':
          new_img = sitk.RescaleIntensity(new_img, 0.0, 1.0)
        else:
          new_img = self.normalize(new_img, self.process_param['norm'],
                                   self.images_raw[self.STD_key].itk[item])

    return new_img

  def post_process(self):
    pass

  def reverse_norm_suv(self, img, item):
    return img * np.max(self.images_raw[self.STD_key][item])

  def get_tags(self, item):
    filepath = self.images_dict['240S']['path'][item].replace(self.IMG_TYPE,
                                                              self.PRO_TYPE)
    return joblib.load(filepath)

  # endregion: data process

  # region: static functions

  @staticmethod
  def load_img(filepath, array=False):
    assert isinstance(filepath, str)
    if not array:
      return sitk.ReadImage(filepath)
    else:
      return sitk.GetArrayFromImage(sitk.ReadImage(filepath))

  @staticmethod
  def percentile(img, percent):
    arr = sitk.GetArrayFromImage(img)
    p = np.percentile(arr, percent)
    arr[arr >= p] = 0
    modified_image = sitk.GetImageFromArray(arr)
    modified_image.CopyInformation(img)
    return modified_image

  @staticmethod
  def crop_by_margin(img, margins):
    ori_size = img.GetSize()
    start_index = [i for i in margins]
    size = [size - 2 * margin for size, margin in zip(ori_size, margins)]
    new_img = sitk.RegionOfInterest(img, size, start_index)
    return new_img

  @staticmethod
  def write_img(img, filepath, refer_img=None):
    assert isinstance(filepath, str)
    if not isinstance(img, sitk.Image):
      img = sitk.GetImageFromArray(img)
    if refer_img is not None:
      img.SetOrigin(refer_img.GetOrigin())
      img.SetDirection(refer_img.GetDirection())
      img.SetSpacing(refer_img.GetSpacing())
    sitk.WriteImage(img, filepath)

  @staticmethod
  def suv_transform(img, tag):
    suv_factor, _, _ = get_suv_factor(tag)
    return sitk.ShiftScale(img, 0, suv_factor)

  @staticmethod
  def normalize(img, type, refer_pet=None):
    if type == 'min-max':
      img = sitk.RescaleIntensity(img, 0.0, 1.0)
    elif type == 'PET':
      assert refer_pet
      refer_max = np.max(sitk.GetArrayFromImage(refer_pet))
      img = sitk.DivideReal(img, float(refer_max))
    return img

  @staticmethod
  def mask2onehot(seg, labels: list):
    onehot = np.zeros_like(seg, dtype=bool)
    onehot[np.isin(seg, labels)] = True
    return onehot

  # endregion: static functions

  # region: special properties

  @property
  def images(self):
    return Dicter(self.data_process, self, key_name='img', name='images')

  @property
  def images_raw(self):
    return Dicter(None, self, key_name='img_itk', name='images')

  @property
  def labels(self):
    return Dicter(self.data_process, self, key_name='img', name='labels')

  @property
  def labels_raw(self):
    return Dicter(None, self, key_name='img_itk', name='labels')

  @property
  def image_keys(self):
    return self._image_keys

  @image_keys.setter
  def image_keys(self, value):
    if value is None:
      return
    for key in value:
      assert key in self.images_dict.keys()
      if key not in self.image_keys and key not in self.label_keys:
        length = len(self.images_dict[key]['path'])
        self.images_dict[key]['img_itk'] = np.array([None] * length)
        self.images_dict[key]['img'] = np.array([None] * length)
    self._image_keys = value

  @property
  def label_keys(self):
    return self._label_keys

  @label_keys.setter
  def label_keys(self, value):
    if value is None:
      return
    for key in value:
      assert key in self.images_dict.keys()
      if key not in self.image_keys and key not in self.label_keys:
        length = len(self.images_dict[key]['path'])
        self.images_dict[key]['img_itk'] = np.array([None]*length)
        self.images_dict[key]['img'] = np.array([None]*length)
        self.images_dict[key]['type'] = None
    self._label_keys = value

  @property
  def STD_key(self):
    return self.img_type['STD'][0]

  # endregion: special properties

  # region: test functions

  def get_stat(self):
    stat_dict = {
      'sex': [],
      'weight': [],
      'age': [],
      'dose': []
    }
    for i in range(len(self.pid)):
      tag = self.get_tags(i)
      stat_dict['sex'].append(tag['PatientSex'])
      stat_dict['weight'].append(int(tag['PatientWeight']))
      stat_dict['age'].append(int(tag['PatientAge'][:-1]))
      stat_dict['dose'].append(int(tag['RadiopharmaceuticalInformationSequence'][0]
                      ['RadionuclideTotalDose'].value//1000000))
    return stat_dict

  @classmethod
  def get_test_sample(cls, csv_path):
    img_dict = {}
    data = np.genfromtxt(csv_path, delimiter=',',
                         dtype=str)
    types = data[0][1:]
    pid = data[1:, 0]
    path_array = data[1:, 1:]

    for i, type_name in enumerate(types):
      img_path = path_array[:, i]
      img_dict[type_name] = {'path': img_path}

    img_type = {
      'CT': ['CT'],
      'PET': ['30G', '40S', '60G-1', '60G-2', '60G-3', '240G'],
      'MASK': ['CT_seg'],
      'STD': ['30G']
    }
    test = cls(img_dict, ['60G-3', '60G-2', 'CT'],
               ['60G-3'], pid,
               img_type=img_type)
    test.rm_void_data()
    return test

  # endregion: test functions

  # region: medical_image compatible


  # region: end






if __name__ == '__main__':
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer
  csv_path = r'../../data/02-RLD-0226/rld_data.csv'

  test = GeneralMI.get_test_sample(csv_path)
  test.process_param['norm'] = 'PET'
  test.process_param['shape'] = [440, 440, 560]
  # test.process_param['percent'] = 99.9
  # test.process_param['ct_window'] = [50, 500]

  # test.LOW_MEM = True
  num = 1
  img1 = test.images['60G-2'][num]
  img2 = test.images['60G-3'][num]

  # onehot = test.mask2onehot(test.images['CT_seg'][0], [5, 10, 11, 12, 13, 14, 51])
  print(img1.shape, img2.shape)

  mi = MedicalImage(test.pid[num], images={'t1': img1, 't2': img2,})
  re = RLDExplorer([mi])
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.sv.set('cmap', 'gist_yarg')
  re.show()

