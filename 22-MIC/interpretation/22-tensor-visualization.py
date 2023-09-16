import sys
import numpy as np

from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path

from mic.mic_agent import MICAgent
from drg.gordonvisualizer import GordonVisualizer



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r'E:\xai-omics\22-MIC\02_fcn\checkpoints\0914_fcn(fcn-4)_Sc_122\0914_fcn(fcn-4)_Sc_122.py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = MICAgent.load()
val_set = val_set.dataset_for_eval
test_set = test_set.dataset_for_eval

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from mic_core import th
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

values = [dataset.features / np.max(dataset.features),
          np.expand_dims(dataset.data_dict['labels'], axis=-1)] + values

patient_ids = dataset.data_dict['patient_ids']
layer_ids = ['input', 'labels'] + [t.name.split('/')[2] for t in tensor_list]

input_data = [[np.transpose(arr[p], axes=(3, 0, 1, 2))
               for arr in values] for p in range(dataset.size)]

dg = GordonVisualizer(
  input_data, patient_ids, layer_ids, title='Tensor Visualizer')
dg.show()

print()
