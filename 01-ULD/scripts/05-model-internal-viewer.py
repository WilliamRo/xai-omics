import sys
import numpy as np
import uld_du as du
from dev.explorers.uld_explorer.uld_explorer import ULDExplorer

from tframe import Predictor
from tframe.utils.file_tools.imp_tools import import_from_path
from xomics import MedicalImage

# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_name = '0831_unet(4-3-3-2-lrelu)test_1-20_w128_s128'

t_file_path = r'../01_unet/checkpoints/' \
              fr'{t_name}/' \
              rf'{t_name}.py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)
# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------

_, _, test_set = du.load_data()

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from uld_core import th
model: Predictor = th.model()

tensor_list = [layer.output_tensor for layer in model.layers
               if 'conv' in layer.full_name]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
values = model.evaluate(tensor_list, test_set, batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
# shape of datas = [1, Layers, Channels, Depth, H, W]
values = [test_set.features / np.max(test_set.features)] + values
data = [np.transpose(arr[0], axes=(3, 0, 1, 2)) for arr in values]
layer_name = ['input'] + [t.name.split('/')[1] for t in tensor_list]

mis = []
for name, value in zip(layer_name, data):
  data_dict = {}
  for i in range(value.shape[0]):
    data_dict[f'{name}-{i}'] = value[i]
  mi = MedicalImage(name, data_dict)
  mis.append(mi)
ue = ULDExplorer(mis)
ue.dv.set('vmin', auto_refresh=False)
ue.dv.set('vmax', auto_refresh=False)
ue.show()

print()
