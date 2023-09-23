import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'pamgau'
id = 6
def model():
  th = core.th
  model = m.get_initial_model()

  sigmas = [int(s) for s in th.sigmas.split(',')]
  N = len(sigmas) + 1
  # Construct DAG
  weights = [m.mu.HyperConv3D(N, kernel_size=int(ks), activation=th.activation)
             for ks in th.archi_string.split('-')]
  weights.append(m.mu.HyperConv3D(N, kernel_size=1, use_bias=th.use_bias,
                                  activation='softmax'))

  vertices = [
    m.GaussianPyramid3D(kernel_size=th.kernel_size, sigmas=sigmas),
    m.mu.Merge.Concat(),
    weights,
    m.WeightedSum(),
  ]
  edges = '1;11;001;0011'
  model.add(m.mu.ForkMergeDAG(vertices, edges))

  return m.finalize(model)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0

  # th.developer_code = 'self2self'
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.dose = ['1-4', '1-10', '1-20', '1-50', '1-100'][4]
  th.data_config = f'epsilon dataset=01-ULD dose={th.dose}'

  th.val_size = 30
  th.test_size = 10
  th.data_shape = [1, 608, 440, 440, 1]
  # th.data_shape = [1, 400, 400, 400, 1]

  th.norm_by_feature = True
  th.sub_indices = [0, 1]

  th.clip_off = False
  # th.suffix = '_01'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '3-3-3'
  th.sigmas = '1,3'
  th.kernel_size = 11
  th.activation = 'relu'
  th.use_bias = True

  th.normalize_energy = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 10
  th.probe_cycle = 0

  th.batch_size = 4
  th.val_batch_size = 1

  th.loss_string = 'rmse'
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.001
  th.val_decimals = 7

  th.developer_code += 'chip'
  # th.developer_code += 'ecc'
  # th.uld_batch_size = 3
  # th.thickness = 10

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, f'({th.archi_string}ks{th.kernel_size})' + f'dose{th.dose}')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
