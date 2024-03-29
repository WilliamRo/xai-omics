from roma import finder

import joblib
import numpy as np
import os
import pydicom
import SimpleITK as sitk


def rd_file_itk(filepath, nii_param=False):
  itk_img = sitk.ReadImage(filepath)
  if nii_param:
    param = {
      # 'size': itk_img.GetSize(),
      'origin': itk_img.GetOrigin(),
      'spacing': itk_img.GetSpacing(),
      'direction': itk_img.GetDirection(),
    }
    return itk_img, param
  return itk_img


def rd_file(filepath, nii_param=False):
  if nii_param:
    itk_img, param = rd_file_itk(filepath, nii_param=nii_param)
    img = sitk.GetArrayFromImage(itk_img)
    return img, param
  else:
    itk_img = rd_file_itk(filepath)
    img = sitk.GetArrayFromImage(itk_img)
    return img


def rd_series_itk(dirpath):
  # fixed: dirpath must be an absolute path
  dirpath = os.path.abspath(dirpath)
  series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dirpath)
  if not series_file_names:
    series_file_names = [os.path.join(dirpath, _) for _ in
                         os.listdir(dirpath)]
  series_reader = sitk.ImageSeriesReader()
  series_reader.SetFileNames(series_file_names)
  image3D = series_reader.Execute()

  return image3D


def rd_series(dirpath, resample=False, refpath=None, refimage=None):
  """
  use pydicom to read image series
  :param dirpath: directory path name
  :return:
  """
  image3D = rd_series_itk(dirpath)
  if resample:
    assert refpath is not None or refimage is not None
    refer_img = rd_series_itk(refpath) if refimage is None else refimage
    image3D = resize_image_itk(image3D, refer_img)

  data = sitk.GetArrayFromImage(image3D)[::-1]

  return data

  # file_paths = finder.walk(dirpath)
  # dcms = [pydicom.read_file(path)
  #         for path in file_paths]
  # dcms = sorted(dcms, key=lambda d: d.InstanceNumber, reverse=True)
  # data = [d.pixel_array for d in dcms]
  # return np.stack(data, axis=0)


def wr_file(arr, pathname, nii_param=None):
  img = sitk.GetImageFromArray(arr)
  if nii_param is not None:
    img.SetOrigin(nii_param['origin'])
    img.SetSpacing(nii_param['spacing'])
    img.SetDirection(nii_param['direction'])
  sitk.WriteImage(img, pathname)


def get_tags(dirpath, suv=True, **kwargs):
  if suv:
    return get_suv_tags(dirpath, **kwargs)
  else:
    return get_all_tags(dirpath, **kwargs)


def get_all_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = pydicom.dcmread(dirpath)
  tags = {}
  for key in data.dir():
    if key == "PixelData":
      continue
    value = getattr(data, key, '')
    tags[key] = value
  return tags


def get_suv_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = pydicom.dcmread(dirpath)
  ST = data.SeriesTime
  AT = data.AcquisitionTime
  PW = data.PatientWeight
  RIS = data.RadiopharmaceuticalInformationSequence[0]
  RST = str(RIS['RadiopharmaceuticalStartTime'].value)
  RTD = str(RIS['RadionuclideTotalDose'].value)
  RHL = str(RIS['RadionuclideHalfLife'].value)
  RS = data.RescaleSlope
  RI = data.RescaleIntercept
  dcm_tag = {
    'ST': ST,
    'AT': AT,
    'PW': PW,
    'RST': RST,
    'RTD': RTD,
    'RHL': RHL,
    'RS': RS,
    'RI': RI
  }
  # for name, key in dcm_tag.items():
  #   print(name, key)
  return dcm_tag


def wr_tags(tags, path, suv=False):
  if suv:
    wr_suv_tags(tags, path)
  else:
    joblib.dump(tags, path)


def wr_suv_tags(tags, path):
  with open(path, 'w+') as f:
    for name, val in tags.items():
      f.write(f'{name},{val}\n')


def rd_tags(path):
  return joblib.load(path)


def npy_save(data, filepath):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  np.save(filepath, data)


def itk_norm(data, vmax, vmin=0):
  filters = sitk.RescaleIntensityImageFilter()
  filters.SetOutputMaximum(vmax)
  filters.SetOutputMinimum(vmin)
  return filters.Execute(data)

def resample_image_by_spacing(image, new_spacing, method=sitk.sitkLinear):
  original_spacing = image.GetSpacing()
  original_size = image.GetSize()

  # 计算缩放因子
  spacing_ratio = [osp / nsp for osp, nsp in
                   zip(original_spacing, new_spacing)]
  new_size = [int(osz * sr) for osz, sr in zip(original_size, spacing_ratio)]

  # 创建一个Resample滤波器
  resample = sitk.ResampleImageFilter()
  resample.SetOutputSpacing(new_spacing)
  resample.SetSize(new_size)
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputDirection(image.GetDirection())

  if method == sitk.sitkNearestNeighbor:
    resample.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
  else:
    resample.SetOutputPixelType(sitk.sitkFloat32)
  resample.SetInterpolator(method)
  # 执行重采样
  resampled_image = resample.Execute(image)

  return resampled_image


def resize_image(original_image, target_size, resamplemethod=sitk.sitkLinear):
  original_spacing = original_image.GetSpacing()
  original_size = original_image.GetSize()

  target_spacing = [original_spacing[0] * (original_size[0] / target_size[0]),
                    original_spacing[1] * (original_size[1] / target_size[1]),
                    original_spacing[2] * (original_size[2] / target_size[2])]

  resampler = sitk.ResampleImageFilter()
  resampler.SetSize(target_size)
  resampler.SetOutputSpacing(target_spacing)
  resampler.SetOutputOrigin(original_image.GetOrigin())
  resampler.SetOutputDirection(original_image.GetDirection())

  if resamplemethod == sitk.sitkNearestNeighbor:
    resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
  else:
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    # 线性插值用于PET/CT/MRI之类的，保存float32
  resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
  resampler.SetInterpolator(resamplemethod)

  resampled_image = resampler.Execute(original_image)
  return resampled_image

def reshape_image(image, target_size):
  original_size = image.GetSize()

  # 计算填充或裁剪的大小差异
  size_diff = [target_size[i] - original_size[i] for i in range(3)]

  # 计算填充或裁剪的边界值
  lower_crop = [-int(size_diff[i] / 2) if size_diff[i] < 0 else 0 for i in range(3)]
  upper_crop = [-size_diff[i] - lower_crop[i] if size_diff[i] < 0 else 0 for i in range(3)]

  lower_pad = [int(size_diff[i] / 2) if size_diff[i] > 0 else 0 for i in range(3)]
  upper_pad = [size_diff[i] - lower_pad[i] if size_diff[i] > 0 else 0 for i in range(3)]

  # 针对奇数尺寸的图像进行对称填充
  # if any(size_diff[i] % 2 != 0 for i in range(3)):
  #   lower_crop = [int(size_diff[i] / 2) for i in range(3)]
  #   upper_crop = [size_diff[i] - lower_crop[i] for i in range(3)]
  #   lower_pad = [lower_crop[i] if size_diff[i] % 2 == 0 else lower_crop[i] + 1
  #                for i in range(3)]
  #   upper_pad = [upper_crop[i] if size_diff[i] % 2 == 0 else upper_crop[i] + 1
  #                for i in range(3)]
  #
  # 对图像进行对称填充
  image = sitk.ConstantPad(image, lower_pad, upper_pad)

  # 对图像进行裁剪

  image = sitk.Crop(image, lower_crop, upper_crop)

  return image



def resize_image_itk(ori_img, target_img=None,
                     size=None, spacing=None, origin=None, direction=None,
                     resamplemethod=sitk.sitkLinear, raw=True):
  """
  用itk方法将原始图像resample到与目标图像一致
  :param ori_img: 原始需要对齐的itk图像
  :param target_img: 要对齐的目标itk图像
  :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
  :return:img_res_itk: 重采样好的itk图像
  使用示范：
  import SimpleITK as sitk
  target_img = sitk.ReadImage(target_img_file)
  ori_img = sitk.ReadImage(ori_img_file)
  img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
  """
  if target_img is not None:
    size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
  else:
    size = size if size is not None else ori_img.GetSize()
    spacing = spacing if spacing is not None else ori_img.GetSpacing()
    origin = origin if origin is not None else ori_img.GetOrigin()
    direction = direction if direction is not None else ori_img.GetDirection()

  if size == ori_img.GetSize():
    if raw:
      return ori_img
    else:
      return sitk.GetArrayFromImage(ori_img)
  # itk的方法进行resample
  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
  # 设置目标图像的信息
  resampler.SetSize(size)  # 目标图像大小
  resampler.SetOutputOrigin(origin)
  resampler.SetOutputDirection(direction)
  resampler.SetOutputSpacing(spacing)
  # 根据需要重采样图像的情况设置不同的type
  if resamplemethod == sitk.sitkNearestNeighbor:
    resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
  else:
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    # 线性插值用于PET/CT/MRI之类的，保存float32
  resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
  resampler.SetInterpolator(resamplemethod)
  itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
  if raw:
    return itk_img_resampled
  else:
    return sitk.GetArrayFromImage(itk_img_resampled)




if __name__ == '__main__':
  pass
  # tags = get_tags(os.path.join(path, s))
  # wr_tags(tags, path+'tags.txt')

