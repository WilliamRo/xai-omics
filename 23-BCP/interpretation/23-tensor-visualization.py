import sys
import numpy as np

from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path

from bcp.bcp_agent import BCPAgent
from drg.gordonvisualizer import GordonVisualizer



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r''
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = BCPAgent.load()

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from mi_core import th
th.val_batch_size = 4
th.eval_batch_size = 4
model: Predictor = th.model()

tensor_list = [layer.output_tensor for layer in model.layers
               if 'conv' in layer.full_name]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
dataset = test_set
values = model.evaluate(tensor_list, dataset, batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
'''
shape of datas = [Patients, Layers, Channels, Depth, H, W]
'''

mi_list = dataset.fetch_data(dataset.size)

features, targets, patient_ids = [], [], []

for mi in mi_list:
  mi.window('ct', th.window[0], th.window[1])
  mi.crop(th.crop_size, random_crop=False)
  mi.normalization(['ct', 'pet'])

  if th.use_pet:
    features.append(
      np.stack([mi.images['ct'], mi.images['pet']], axis=-1))
  else:
    features.append(np.expand_dims(mi.images['ct'], axis=-1))

  targets.append(np.expand_dims(mi.labels['label-0'], axis=-1))
  patient_ids.append(mi.key)

layer_ids = ['input', 'label'] + [t.name.split('/')[1]
                                  for t in tensor_list]

values = [np.array(features), np.array(targets)] + values

input_data = [[np.transpose(arr[p], axes=(3, 0, 1, 2))
               for arr in values] for p in range(dataset.size)]

dg = GordonVisualizer(
  input_data, patient_ids, layer_ids, title='Tensor Visualizer')
dg.show()
