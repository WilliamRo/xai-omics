import os
import numpy as np
from pydicom import dcmread
import SimpleITK as sitk
from skimage import exposure


def normalize(a):
  a = (a - np.min(a)) / np.max(a)
  return a


def rd_file(filepath):
  itk_img = sitk.ReadImage(filepath)
  img = sitk.GetArrayFromImage(itk_img)
  return img


def get_tags(dirpath, isSeries=False):
  if isSeries:
    dirpath = os.path.join(dirpath, os.listdir(dirpath)[0])
  data = dcmread(dirpath)
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


def wr_tags(tags, path):
  with open(path,'w+') as f:
    for name, val in tags.items():
      f.write(f'{name},{val}\n')


def dicom_time(t):
  t = str(t)
  if len(t) == 5:
    t = '0' + t
  h_t = float(t[0:2])
  m_t = float(t[2:4])
  s_t = float(t[4:6])
  return h_t * 3600 + m_t * 60 + s_t


def calc_SUV(data, tags, norm=False):
  decay_time = dicom_time(tags['ST']) - dicom_time(tags['RST'])
  decay_dose = float(tags['RTD']) * pow(2, -float(decay_time) / float(tags['RHL']))
  SUVbwScaleFactor = (1000 * float(tags['PW'])) / decay_dose
  if norm:
    PET_SUV = (data * float(tags['RS']) + float(tags['RI'])) * SUVbwScaleFactor
  else:
    PET_SUV = data * SUVbwScaleFactor
  return PET_SUV


def rd_series(dirpath):
  series_reader = sitk.ImageSeriesReader()
  filenames = series_reader.GetGDCMSeriesFileNames(dirpath)
  series_reader.SetFileNames(filenames)
  data = series_reader.Execute()
  images = sitk.GetArrayFromImage(data)
  return images


def wr_file(arr, pathname):
  sitk.WriteImage(arr, pathname)


def npy_save(data, filepath):
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  np.save(filepath, data)

# for uld train raw data
def rd_uld_train(datapath, subject, dose="Full_dose"):
  patients = os.listdir(os.path.join(datapath, subject))
  images = []
  tags = []

  for patient in patients:
    dirpath = os.path.join(datapath, subject, patient, dose)
    # img = normalize(rd_series(dirpath))
    img = rd_series(dirpath)
    tag_dict = get_tags(dirpath, isSeries=True)
    # if img.shape[0] % 2 != 0:
    #   img = img[1:]
    # cut = (img.shape[0] - 608) // 2
    # img = img[cut:-cut]
    img = img.reshape((1,) + img.shape + (1,))

    images.append(img)
    tags.append(tag_dict)

  # results = np.concatenate([arr[np.newaxis] for arr in images], axis=0)
  # results = results.reshape(results.shape + (1,))

  return images, tags


def gen_npy(path, n_path):
  subjects = os.listdir(path)[:-1]
  count = 0
  for subject in subjects:
    count += 1
    print(f"({count}/{len(subjects)}) Reading {subject}...")
    c1 = 0
    for dose in doses:
      c1 += 1
      data, tags = rd_uld_train(path, subject, dose=dose)
      num = int(subject[8:].split('-')[0])
      print(f"..({c1}/{len(doses)}) Reading {dose}")
      dose = dose[:-5]
      c2 = 0
      for arr, tag in zip(data, tags):
        c2 += 1
        # print(arr.shape, tag)
        npypath = os.path.join(n_path,
                               f'subject{num}', f'subject{num}_{dose}.npy')
        tagpath = os.path.join(n_path,
                               f'subject{num}', f'tags_subject{num}_{dose}.txt')
        npy_save(arr, npypath)
        print(f"....({c2}/{len(data) * 2}) Save numpy data subject{num} {dose}")
        wr_tags(tag, tagpath)
        c2 += 1
        print(f"....({c2}/{len(data) * 2}) Save tags data subject{num} {dose}")
        num += 1


# for uld test raw data
def rd_uld_test(dirpath, datanum=1):
  files = os.listdir(dirpath)[:datanum]
  arr = []
  for file in files:
    filepath = os.path.join(dirpath, file)
    arr.append(rd_file(filepath))
  results = np.stack(arr)
  return results




if __name__ == '__main__':
  doses = [
    'Full_dose',
    '1-2 dose',
    '1-4 dose',
    '1-10 dose',
    '1-20 dose',
    '1-50 dose',
    '1-100 dose',
    ]
  path = "../../data/01-ULD-RAW/"
  n_path = "../../data/01-ULD/"
  gen_npy(path, n_path)
  # tags = get_tags(os.path.join(path, s))
  # wr_tags(tags, path+'tags.txt')

