import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'pam'
id = 6
def model():
  th = core.th
  model = m.get_initial_model()

  # Prepare filter generator
  fg = m.gen_ecc_filter if 'ecc' in th.developer_code else None

  # Construct DAG
  weights = [m.mu.HyperConv3D(2, kernel_size=int(ks), activation=th.activation)
       for ks in th.archi_string.split('-')]
  weights.append(m.mu.HyperConv3D(2, kernel_size=1, activation='softmax'))

  vertices = [
    m.mu.HyperConv3D(1, kernel_size=th.kernel_size, filter_generator=fg),
    m.mu.Merge.Concat(),
    weights,
    m.WeightedSum(),
  ]
  edges = '1;11;100;0011'
  model.add(m.mu.ForkMergeDAG(vertices, edges))

  return m.finalize(model)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.dose = ['1-2', '1-4', '1-10', '1-20', '1-50', '1-100'][4]
  th.data_config = f'epsilon dataset=01-ULD dose={th.dose}'

  th.val_size = 1
  th.test_size = 1
  th.data_shape = [1, 800, 256, 256, 1]
  # th.data_shape = [1, 400, 400, 400, 1]

  th.norm_by_feature = True

  # th.sub_indices = [0]
  # th.slice_range = [160, 190]

  th.sub_indices = [1]
  th.slice_range = [60, 100]
  # th.suffix = '_01'
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  # th.suffix = ''

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '3-1-1'
  th.kernel_size = 3
  th.activation = 'relu'

  th.normalize_energy = 0
  # th.developer_code += 'ecc'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.train = 0
  th.overwrite = 1

  th.epoch = 1000
  th.early_stop = True
  th.patience = 10
  th.probe_cycle = th.updates_per_round

  th.batch_size = 1
  th.val_batch_size = 1

  th.loss_string = ['mae', 'rmse'][1]
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.001
  th.val_decimals = 7

  th.developer_code += 'chip'
  # th.developer_code += 'ecc'
  th.uld_batch_size = 3
  th.thickness = 10
  th.updates_per_round = 20

  if not th.train:
    th.visible_gpu_id = -1
    th.show_weight_map = True
    # th.data_shape = [1, 600, 400, 400, 1]
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, f'({th.archi_string}ks{th.kernel_size})-{th.developer_code}' +
    f'dose{th.dose}')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
