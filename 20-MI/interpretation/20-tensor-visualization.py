import sys
import numpy as np

from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path

from mi.mi_agent import MIAgent
from drg.gordonvisualizer import GordonVisualizer



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r'E:\xai-omics\10-MI\01_unet\checkpoints\0727_unet(2-3-2-2-lrelu)\0727_unet(2-3-2-2-lrelu).py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = MIAgent.load()

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from mi_core import th
model: Predictor = th.model()

tensor_list = [layer.output_tensor for layer in model.layers
               if 'conv' in layer.full_name]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
values = model.evaluate(tensor_list, train_set, batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
# shape of datas = [1, Layers, Channels, Depth, H, W]
values = [train_set.features / np.max(train_set.features)] + values
data = [[np.transpose(arr[0], axes=(3, 0, 1, 2)) for arr in values]]
layer_name = ['input'] + [t.name.split('/')[1] for t in tensor_list]

dg = GordonVisualizer(data, layer_name, title='Tensor Visualizer')
dg.show()

print()