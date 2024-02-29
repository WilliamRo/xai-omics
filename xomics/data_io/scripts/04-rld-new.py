import os
import SimpleITK as sitk

from roma import console
from xomics.data_io.utils.raw_rw import rd_series_itk, get_tags, wr_tags


def get_src_path(src_dir):
  patients_dirs = sorted(os.listdir(src_dir))
  src_path = {}
  for patient_dir in patients_dirs:
    patient_path = os.path.join(src_dir, patient_dir)
    tmp_dir = os.listdir(patient_path)[0]

    patient_path = os.path.join(patient_path, tmp_dir)
    img_dirs = os.listdir(patient_path)

    img_path = []
    for img_dir in img_dirs:
      if 'BH' in img_dir:
        continue
      img_path.append(os.path.join(patient_path, img_dir))
    src_path[patient_dir] = img_path

  return src_path


def rld2nii(src_dir, tgt_dir):
  """
  :param src_dir: must be an absolute path
  :param tgt_dir: must be an absolute path
  :return:
  """
  console.supplement("start convert raw to nii...")
  console.supplement(f"src_dir: {src_dir}, tgt_dir: {tgt_dir}", level=2)
  src_path = get_src_path(src_dir)

  patient_len = len(src_path)
  img_len = sum([len(_) for _ in src_path.values()])
  sum_count = 1
  p_count = 1
  for patient, imgs_path in src_path.items():
    console.supplement(f"[{p_count}/{patient_len}] start process {patient}...")
    pid = patient.split('_')[-1]
    save_dir = os.path.join(tgt_dir, pid)

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
      with open(os.path.join(save_dir, patient), 'w') as _:
        pass

    img_count = 1
    for img_path in imgs_path:
      img_name = os.path.basename(img_path)
      console.supplement(f"[{img_count}/{len(imgs_path)}] processing {img_name}",
                         level=2)
      console.print_progress(sum_count, img_len)
      sum_count += 1
      img_count += 1
      save_path = os.path.join(save_dir, img_name[:-5])

      img = rd_series_itk(img_path)
      sitk.WriteImage(img, save_path + '.nii.gz')

      tag = get_tags(img_path, suv=False, isSeries=True)
      wr_tags(tag, save_path + '.pkl')
    console.clear_line()
    p_count += 1
    pass



if __name__ == '__main__':
  # src_dir = r'D:/projects/xai-omics/data/02-RLD-RAW-0226/'
  src_dir = r'D:/projects/xai-omics/data/02-RLD-RAW/'
  tgt_dir = r'D:/projects/xai-omics/data/02-RLD/'

  rld2nii(src_dir, tgt_dir)

  # current_dir = r'\\192.168.5.99\xai\xai-omics\data\02-RLD-RAW'
  # file_list = get_src_path(current_dir)
  pass
