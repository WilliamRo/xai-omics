import sys
import numpy as np

from scipy.ndimage import zoom
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

# get the feature map
tensor = [[layer.output_tensor for layer in model.layers
           if 'conv' in layer.full_name][-1]] + [model.layers[-1].output_tensor]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
dataset = test_set
values = model.evaluate(tensor, dataset, batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Data processing
# -----------------------------------------------------------------------------
'''
shape of datas = [Patients, Layers, Channels, Depth, H, W]
'''
patient_ids = dataset.data_dict['patient_ids']
layer_ids = ['feature map', 'outputs', 'input']

input_data = [values[0], np.expand_dims(dataset.data_dict['labels'], axis=-1)]
input_data = input_data + [dataset.data_dict['features']]
input_data = [[np.transpose(arr[p], axes=(3, 0, 1, 2))
               for arr in input_data] for p in range(dataset.size)]

# normalization and resize
for patient in input_data:
  mean = patient[0].mean(axis=(1, 2, 3), keepdims=True)
  std = patient[0].std(axis=(1, 2, 3), keepdims=True)
  patient[0] = (patient[0] - mean) / std
  patient[0] = np.array([zoom(c, (1, 8, 8), order=3) for c in patient[0]])

# -----------------------------------------------------------------------------
# 6. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
cancer_type = ['BA', 'MIA']
indice_prediction = np.argmax(values[1], axis=-1)
indices_target = np.argmax(dataset.targets, axis=-1)
print('\n')
for i in range(dataset.size):
  print(f'Number: {i} -- '
        f'Ground Truth:{cancer_type[indices_target[i]]} -- '
        f'Prediction: {cancer_type[indice_prediction[i]]} -- '
        f'BA: {round(values[1][i][0] * 100, 2)} -- '
        f'MIA: {round(values[1][i][1] * 100, 2)}')


dg = GordonVisualizer(input_data, patient_ids, layer_ids,
                      title='Tensor Visualizer')
dg.show()

print()