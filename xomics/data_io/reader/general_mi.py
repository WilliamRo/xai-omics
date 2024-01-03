import copy
import numpy as np
import SimpleITK as sitk
import joblib


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
    self._name = name
    self._key = key
    self._data = self._obj.images_dict[key]
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
      for i in item:
        data.append(self.get_data(i))
      return data
    elif isinstance(item, slice):
      start = item.start if item.start else 0
      stop = item.stop if item.stop else len(self)
      step = item.step if item.step else 1
      iterator = iter(range(start, stop, step))
      data = []
      for i in iterator:
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
        assert self._obj.images_raw.itk[item]
      elif self._name == 'labels':
        assert self._obj.labels_raw.itk[item]
      self.data[self._img_key][item] = self.process_func(self.data, self._key, item)
    return sitk.GetArrayFromImage(self.data[self._img_key][item])

  @property
  def itk(self):
    return ItkIndexer(self.process_func, self._obj, self._name,
                      self._key, self._img_key)


class GeneralMI:

  def __init__(self, images_dict, image_key=None, label_key=None,
               pid=None, process_param=None):
    assert image_key in images_dict.keys() or image_key is None
    assert label_key in images_dict.keys() or label_key is None
    self.TYPE = 'nii.gz'
    self.pid = pid

    self.images_dict = images_dict
    self.image_key = image_key
    self.label_key = label_key

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
      if key == self.image_key:
        image_dict[key]['img_itk'] = self.images_dict[key]['img_itk'][item]
      elif key == self.label_key:
        image_dict[key]['img_itk'] = self.images_dict[key]['img_itk'][item]

    return GeneralMI(image_dict, image_key=self.image_key, label_key=self.label_key,
                     pid=pid, process_param=self.process_param)

  def index(self, pid):
    return np.where(self.pid == pid)

  def raw_process(self, img, key, item):
    new_img = img['img_itk'][item]
    if 'CT' not in key:
      new_img = GeneralMI.suv_transform(new_img,
                                        img['path'][item].replace(self.TYPE, 'pkl'))
      if 'CT' in self.image_key:
        new_img = resize_image_itk(new_img, self.images_raw.itk[item])
      elif 'CT' in self.label_key:
        new_img = resize_image_itk(new_img, self.labels_raw.itk[item])
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
    return ImgIndexer(self.data_process, self, img_key='img',
                      name='images', key=self.image_key)

  @property
  def images_raw(self):
    return ImgIndexer(self.raw_process, self, img_key='img_itk',
                      name='images', key=self.image_key)

  @property
  def labels(self):
    return ImgIndexer(self.data_process, self, img_key='img',
                      name='labels', key=self.label_key)

  @property
  def labels_raw(self):
    return ImgIndexer(self.raw_process, self, img_key='img_itk',
                      name='labels', key=self.label_key)

  @property
  def image_key(self):
    return self._image_key

  @image_key.setter
  def image_key(self, value):
    self._image_key = value
    if value is None:
      return
    length = len(self.images_dict[self._image_key]['path'])
    self.images_dict[self._image_key]['img_itk'] = [None]*length
    self.images_dict[self._image_key]['img'] = [None]*length

  @property
  def label_key(self):
    return self._label_key

  @label_key.setter
  def label_key(self, value):
    self._label_key = value
    if value is None:
      return
    length = len(self.images_dict[self._label_key]['path'])
    self.images_dict[self._label_key]['img_itk'] = [None]*length
    self.images_dict[self._label_key]['img'] = [None]*length





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
  test = GeneralMI(img_dict, '30G', '240G', pid)

  shape = test.images[0].shape + (1,)
  img = test.images[0]
  img2 = test.labels[0]
  print(img.shape, img2.shape)
  mi = MedicalImage('test', images={'t1': img, 't2': img2})
  re = RLDExplorer([mi])
  re.sv.set('vmin', auto_refresh=False)
  re.sv.set('vmax', auto_refresh=False)
  re.show()

