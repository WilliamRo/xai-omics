import copy
import numpy as np
import SimpleITK as sitk
import joblib

from roma import console
from xomics import MedicalImage
from xomics.data_io.utils.preprocess import get_suv_factor
from xomics.data_io.utils.raw_rw import reshape_image, resize_image_itk


#  images_dict =
#  {'type1':
#    {'path': [path], 'itk_img': [image_itk], 'img':[img_itk]}
#  }


class Indexer:

  def __init__(self, obj, name='noname', types=None, key_name=None):
    self._obj: GeneralMI = obj
    assert name in ['images', 'labels', 'noname']
    self.name = name
    self.type_name = types
    self._data = self._obj.images_dict[types] if types else None
    self.key_name = key_name

  @property
  def data(self):
    return self._data

  @data.getter
  def data(self):
    self._data = self._obj.images_dict[self.type_name]
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
      iterator = iter(range(start, stop, step))
      data = []
      for num, i in enumerate(iterator):
        console.print_progress(num, int((stop-start)/step))
        data.append(self.get_data(i))
      console.clear_line()
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

  def __init__(self, process_func=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.PROCESS_FUNC = process_func

  def get_data(self, item):
    if self.data[self.key_name][item] is None:
      self.data[self.key_name][item] = GeneralMI.load_img(self.data['path'][item])
      if self.PROCESS_FUNC is not None:
        self.data[self.key_name][item] = self.PROCESS_FUNC(self.data[self.key_name][item],
                                                           self.data,
                                                           self.type_name,
                                                           self.name, item,
                                                           self.type)
    tmp = self.data[self.key_name][item]
    if self._obj.LOW_MEM:
      self.data[self.key_name][item] = None
    return tmp

  @property
  def type(self):
    for key, item in self._obj.img_type.items():
      if self.type_name in item:
        return key
    return 'Unknown'



class ImgIndexer(ItkIndexer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_data(self, item):
    if self.data[self.key_name][item] is None:
      if self.name == 'images':
        img = self._obj.images_raw[self.type_name].itk[item]
      elif self.name == 'labels':
        img = self._obj.labels_raw[self.type_name].itk[item]
      self.data[self.key_name][item] = self.PROCESS_FUNC(img, self.data,
                                                         self.type_name, self.name,
                                                         item, self.type)
    tmp = self.data[self.key_name][item]
    if self._obj.LOW_MEM:
      self.data[self.key_name][item] = None
    return sitk.GetArrayFromImage(tmp)

  @property
  def itk(self):
    return ItkIndexer(self._obj.raw_process, self._obj, name=self.name,
                      types=self.type_name, key_name='img_itk')


class Dicter(ImgIndexer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __getitem__(self, item):
    assert isinstance(item, (str, np.str_, int))
    if isinstance(item, int):
      if self.name == 'images':
        assert item < len(self._obj.image_keys)
        return ImgIndexer(self.PROCESS_FUNC, self._obj, key_name=self.key_name,
                          name=self.name, types=self._obj.image_keys[item])
      elif self.name == 'labels':
        assert item < len(self._obj.label_keys)
        return ImgIndexer(self.PROCESS_FUNC, self._obj, key_name=self.key_name,
                          name=self.name, types=self._obj.label_keys[item])

    item = str(item)
    if self.name == 'images':
      assert item in self._obj.image_keys
    elif self.name == 'labels':
      assert item in self._obj.label_keys
    return ImgIndexer(self.PROCESS_FUNC, self._obj, key_name=self.key_name,
                      name=self.name, types=item)



class GeneralMI:
  """

  """
  def __init__(self, images_dict, image_keys=None, label_keys=None,
               pid=None, process_param=None, img_type=None):
    self.IMG_TYPE = 'nii.gz'
    self.PRO_TYPE = 'pkl'
    self.LOW_MEM = False
    self.pid = pid

    self._image_keys, self._label_keys = [], []
    self.images_dict = images_dict
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

  def __len__(self):
    return len(self.pid)

  def __getitem__(self, item):
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

  def pre_process(self, img, data, key_name, name, item, type):
    new_img = img
    if type == 'MASK':
      pass
    elif type == 'PET':
      new_img = self.suv_transform(new_img, data['path'][item].
                                   replace(self.IMG_TYPE, self.PRO_TYPE))
      if self.process_param.get('percent'):
        new_img = self.percentile(new_img, self.process_param['percent'])
    elif type == 'CT':
      if self.process_param.get('ct_window'):
        wc = self.process_param['ct_window'][0]
        wl = self.process_param['ct_window'][1]
        new_img = sitk.IntensityWindowing(new_img, wc - wl/2, wc + wl/2, 0, 255)
      else:
        new_img = sitk.RescaleIntensity(new_img, 0, 255)
    if type != 'PET':
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

  def process(self, img, data, key_name, name, item, type):
    new_img: sitk.Image = img
    if 'MASK' == type:
      pass
    elif 'CT' == type:
      if self.process_param.get('norm'):
        new_img = sitk.RescaleIntensity(new_img, 0.0, 1.0)
    elif 'PET' == type:
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

  def clean_mem(self):
    for key in self.images_dict.keys():
      length = len(self.images_dict[key]['path'])
      self.images_dict[key]['img_itk'] = np.array([None] * length)
      self.images_dict[key]['img'] = np.array([None] * length)

  def reverse_norm_suv(self, img, item):
    return img * np.max(self.images_raw[self.STD_key][item])

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
  def load_img(filepath):
    assert isinstance(filepath, str)
    return sitk.ReadImage(filepath)

  @staticmethod
  def suv_transform(img, path):
    tag = joblib.load(path)
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

  @property
  def images(self):
    return Dicter(self.data_process, self, key_name='img', name='images')

  @property
  def images_raw(self):
    return Dicter(self.raw_process, self, key_name='img_itk', name='images')

  @property
  def labels(self):
    return Dicter(self.data_process, self, key_name='img', name='labels')

  @property
  def labels_raw(self):
    return Dicter(self.raw_process, self, key_name='img_itk', name='labels')

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






if __name__ == '__main__':
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer

  img_dict = {}
  data = np.genfromtxt('../../data/02-RLD/rld_data.csv', delimiter=',', dtype=str)
  types = data[0][1:]
  pid = data[1:, 0]
  path_array = data[1:, 1:]

  for i, type_name in enumerate(types):
    img_path = path_array[:, i]
    img_dict[type_name] = {'path': img_path}

  img_type = {
    'CT': ['CT'],
    'PET': ['240G'],
    'MASK': ['CT_seg'],
    'STD': ['30G']
  }

  test = GeneralMI(img_dict, ['30G', 'CT', 'CT_seg'], ['240G'], pid, img_type=img_type)
  # test.process_param['norm'] = 'PET'
  test.process_param['shape'] = [440, 440, 256]
  test.process_param['percent'] = 99.9
  # test.process_param['ct_window'] = [50, 500]

  # test.LOW_MEM = True
  img = test.images['CT'][0]
  img2 = test.labels['240G'][0]

  print(img.shape, img2.shape)

  mi = MedicalImage('test', images={'t1': img, 't2': img2, 'seg':test.images[1][0]})
  re = RLDExplorer([mi])
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.show()

