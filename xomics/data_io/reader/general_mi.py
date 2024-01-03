import copy
import numpy as np
import SimpleITK as sitk
import joblib

from roma import console
from xomics import MedicalImage
from xomics.data_io.utils.preprocess import get_suv_factor
from xomics.data_io.utils.raw_rw import resize_image, resize_image_itk


#  images_dict =
#  {'type1':
#    {'path': [path], 'itk_img': [image_itk], 'img':[img_itk]}
#  }


class Indexer:

  def __init__(self, obj, name='noname', key=None, img_key=None):
    self._obj: GeneralMI = obj
    assert name in ['images', 'labels', 'noname']
    self._name = name
    self._key = key
    self._data = self._obj.images_dict[key] if key else None
    self._img_key = img_key

  @property
  def data(self):
    return self._data

  @data.getter
  def data(self):
    self._data = self._obj.images_dict[self._key]
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
    if isinstance(item, int):
      return self.get_data(item)
    elif isinstance(item, tuple):
      return self.get_data(int(item[0][0]))
    elif isinstance(item, list) or isinstance(item, np.ndarray):
      data = []
      for num, i in enumerate(item):
        console.print_progress(num, len(item))
        data.append(self.get_data(i))
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
    self.process_func = process_func

  def get_data(self, item):
    if self.data[self._img_key][item] is None:
      self.data[self._img_key][item] = GeneralMI.load_img(self.data['path'][item])
      if self.process_func is not None:
        self.data[self._img_key][item] = self.process_func(self.data, self._key, item)
    return self.data[self._img_key][item]


class ImgIndexer(ItkIndexer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_data(self, item):
    if self.data[self._img_key][item] is None:
      if self._name == 'images':
        assert self._obj.images_raw[self._key].itk[item]
      elif self._name == 'labels':
        assert self._obj.labels_raw[self._key].itk[item]
      self.data[self._img_key][item] = self.process_func(self.data, self._key, item)
    return sitk.GetArrayFromImage(self.data[self._img_key][item])

  @property
  def itk(self):
    return ItkIndexer(self.process_func, self._obj, self._name,
                      self._key, self._img_key)


class Dicter(ImgIndexer):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __getitem__(self, item):
    assert isinstance(item, str) or isinstance(item, np.str_) or isinstance(item, int)
    if isinstance(item, int):
      if self._name == 'images':
        assert item < len(self._obj.image_keys)
        return ImgIndexer(self.process_func, self._obj, img_key=self._img_key,
                          name=self._name, key=self._obj.image_keys[item])
      elif self._name == 'labels':
        assert item < len(self._obj.label_keys)
        return ImgIndexer(self.process_func, self._obj, img_key=self._img_key,
                          name=self._name, key=self._obj.label_keys[item])

    item = str(item)
    if self._name == 'images':
      assert item in self._obj.image_keys
    elif self._name == 'labels':
      assert item in self._obj.label_keys
    return ImgIndexer(self.process_func, self._obj, img_key=self._img_key,
                      name=self._name, key=item)



class GeneralMI:

  def __init__(self, images_dict, image_keys=None, label_keys=None,
               pid=None, process_param=None):
    assert all(key in images_dict.keys() for key in image_keys)
    assert all(key in images_dict.keys() for key in label_keys)
    self.TYPE = 'nii.gz'
    self.pid = pid

    self._image_keys, self._label_keys = [], []
    self.images_dict = images_dict
    self.image_keys = image_keys
    self.label_keys = label_keys

    self.data_process = self.process
    self.process_param = {
      'ct_window': None,
      'norm': None,  # only min-max,
      'shape': None,  # [320, 320, 240]
      'crop': None,  # [30, 30, 10]
      'clip': None,  # [1, None]
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
                     pid=pid, process_param=self.process_param)

  def index(self, pid):
    return np.where(self.pid == pid)

  def raw_process(self, img, key, item):
    new_img = img['img_itk'][item]
    if 'CT' not in key:
      new_img = GeneralMI.suv_transform(new_img,
                                        img['path'][item].replace(self.TYPE, 'pkl'))
      if 'CT' in self.image_keys:
        new_img = resize_image_itk(new_img, self.images_raw['CT'].itk[item])
      elif 'CT' in self.label_keys:
        new_img = resize_image_itk(new_img, self.labels_raw['CT'].itk[item])
    else:
      if self.process_param['ct_window'] is not None:
        wc = self.process_param['ct_window'][0]
        wl = self.process_param['ct_window'][1]
        new_img = sitk.IntensityWindowing(img, wc - wl/2, wc + wl/2, 0, 255)
    if self.process_param.get('crop'):
      new_img = GeneralMI.crop_by_margin(new_img, self.process_param['crop'])
    if self.process_param.get('shape'):
      new_img = resize_image(new_img, self.process_param['shape'])
    return new_img

  def process(self, img, key, item):
    new_img: sitk.Image = img['img_itk'][item]

    if self.process_param.get('clip'):
      clip = self.process_param['clip']
      if clip[0] is None:
        clip[0] = np.min(sitk.GetArrayFromImage(new_img))
      elif clip[1] is None:
        clip[1] = np.max(sitk.GetArrayFromImage(new_img))
      new_img = sitk.IntensityWindowing(new_img)
    if self.process_param.get('norm') == 'min-max':
      new_img = sitk.RescaleIntensity(new_img,
                                      outputMinimum=0.0, outputMaximum=1.0)
    return new_img

  def post_process(self):
    pass

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
    return sitk.ShiftScale(img, 100.0, suv_factor)

  @property
  def images(self):
    return Dicter(self.data_process, self, img_key='img', name='images')

  @property
  def images_raw(self):
    return Dicter(self.raw_process, self, img_key='img_itk', name='images')

  @property
  def labels(self):
    return Dicter(self.data_process, self, img_key='img', name='labels')

  @property
  def labels_raw(self):
    return Dicter(self.raw_process, self, img_key='img_itk', name='labels')

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
        self.images_dict[key]['img_itk'] = [None] * length
        self.images_dict[key]['img'] = [None] * length
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
        self.images_dict[key]['img_itk'] = [None]*length
        self.images_dict[key]['img'] = [None]*length
    self._label_keys = value





if __name__ == '__main__':
  from dev.explorers.rld_explore.rld_explorer import RLDExplorer

  img_dict = {}
  data = np.genfromtxt('../../../data/02-RLD/rld_data.csv', delimiter=',', dtype=str)
  types = data[0][1:]
  pid = data[1:, 0]
  path_array = data[1:, 1:]

  for i, type_name in enumerate(types):
    img_path = path_array[:, i]
    img_dict[type_name] = {'path': img_path}
  test = GeneralMI(img_dict, ['30G', 'CT'], ['240G'], pid)

  shape = test.images['30G'][0].shape + (1,)
  img = test.images['CT'][0]
  t = test.images[0][slice(0, 10, 3)]
  img2 = test.labels[0][0]
  print(img.shape, img2.shape)
  mi = MedicalImage('test', images={'t1': img, 't2': img2})
  re = RLDExplorer([mi])
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.show()

