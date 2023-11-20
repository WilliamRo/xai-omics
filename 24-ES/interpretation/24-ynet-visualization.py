import sys
import numpy as np
import os

from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path
from es.es_agent import ESAgent
from drg.gordonvisualizer import GordonVisualizer
from tools import image_processing
from xomics import MedicalImage



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r'E:\xai-omics\24-ES\03_ynet\checkpoints\1117_ynet(8-5-2-3-relu-mp)_Sc_10\1117_ynet(8-5-2-3-relu-mp)_Sc_10.py'
t_file_path = r'E:\xai-omics\24-ES\03_ynet\checkpoints\1120_ynet(8-5-2-3-relu-mp)_Sc_1\1120_ynet(8-5-2-3-relu-mp)_Sc_1.py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = ESAgent.load()

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from es_core import th
th.val_batch_size = 2
th.eval_batch_size = 2
model: Predictor = th.model()

tensor_list = [context.depot['region_layer'].output_tensor,
               context.depot['lesion_layer'].output_tensor,
               model.layers[-1].output_tensor]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
datasets = train_set, val_set, test_set
for dataset in datasets:
  values = model.evaluate(tensor_list, dataset, batch_size=1, verbose=True)

  # -----------------------------------------------------------------------------
  # 5. Visualize tensor in Pictor
  # -----------------------------------------------------------------------------
  '''
  shape of datas = [Patients, Layers, Channels, Depth, H, W]
  '''
  regions = np.squeeze(values[0])
  lesions = np.squeeze(values[1])
  predictions = np.squeeze(values[2])

  assert regions.shape == lesions.shape == predictions.shape

  features = np.squeeze(dataset.features)
  targets = np.squeeze(dataset.targets)
  pids = dataset.data_dict['pids'].tolist()
  original_regions = np.squeeze(dataset.data_dict['region_mask'])
  classes = np.squeeze(dataset.data_dict['class'])

  save_dir = r'E:\xai-omics\data\02-PET-CT-Y1\results\03-ynet\mi'
  model_file = t_file_path.split('\\')[-1].split('.py')[0]
  save_dir = os.path.join(save_dir, model_file)
  if not os.path.exists(save_dir): os.mkdir(save_dir)

  mi_list = []
  for f, l, r, p, pid, cls, o_r, o_l in zip(
      features, lesions, regions, predictions, pids, classes, original_regions, targets):
    ct, pet = f[..., 0], f[..., 1]
    basis = [o_l, o_r] if np.array_equal(cls, [0, 1]) else [o_l]
    ct, pet, o_l = image_processing.crop_3d(
      [ct, pet, o_l], [128, 256, 256], False, basis)

    images = {'ct': ct, 'pet': pet}
    labels = {'Lesion-Ground Truth': o_l, 'Prediction': p, 'P-Region': r}

    acc = dataset.dice_accuarcy(ground_truth=o_l, prediction=p)
    key = pid + f' --- Dice Acc: {round(acc, 2)}'
    print(key)

    mi: MedicalImage = MedicalImage(
      images=images, labels=labels, key=key)
    mi.save(os.path.join(save_dir, pid + '.mi'))
    # mi_list.append(mi)

  # dataset.visulization(mi_list)


