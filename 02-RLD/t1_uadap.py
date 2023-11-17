
import rld_core as core
import rld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'uadap'
id = 3
def model():
  th = core.th
  model = m.get_initial_model()

  sigmas = [int(s) for s in th.archi_string.split('-')]
  N = len(sigmas) + 2
  # Construct DAG
  weights = [
    m.mu.HyperConv3D(N, kernel_size=int(ks), activation=th.activation)
    for ks in th.archi_string.split('-')]
  # if th.beta > 0: weights.insert(0, m.Highlighter(th.beta))
  weights.append(m.mu.HyperConv3D(N, kernel_size=1, use_bias=th.use_bias,
                                  activation='softmax'))

  unet = m.get_unet_list('4-3-3-2-' + th.activation)
  unet.append(m.mu.HyperConv3D(filters=1, kernel_size=1))
  unet.append(m.Clip(0, 1.0))

  vertices = [
    m.GaussianPyramid3D(kernel_size=th.kernel_size, sigmas=sigmas),
    m.mu.Merge.Concat(),
    unet,
    m.mu.Merge.Concat(),
    weights,
    m.WeightedSum(),
  ]
  edges = '1;11;100;0011;00001;000011'
  model.add(m.mu.ForkMergeDAG(vertices, edges))
  return m.finalize(model)


def main(_):
  console.start('{} on PET/CT reconstruct task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------

  th.data_config = fr'alpha dataset=02-RLD'

  th.val_size = 5
  th.test_size = 2

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128
  th.data_shape = [256, 440, 440]
  th.data_set = [1, 3]
  th.data_margin = [10, 0, 0]

  th.noCT = True
  if th.noCT:
    th.input_shape[-1] = 1
  # th.use_suv = False

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.suffix += f'_w{th.window_size}_s{th.slice_size}'


  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.kernel_size = 3
  th.activation = 'lrelu'
  th.archi_string = '1-3'
  th.use_bias = True

  th.use_sigmoid = False
  th.clip_off = False
  th.output_conv = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = th.updates_per_round

  th.batch_size = 4
  th.batchlet_size = 1
  th.val_batch_size = 2

  th.buffer_size = 6

  th.loss_string = 'nrmse'
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.show_weight_map = True

  if th.use_suv:
    th.suffix += '_suv'
  th.suffix += '_noCT' if th.noCT else ''
  th.suffix += '_30Gto240G'
  th.suffix += f'_{th.opt_str}'
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
