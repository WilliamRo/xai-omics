import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'amen'
id = 5
def model():
  th = core.th
  model = m.get_initial_model()

  v1 = []
  for str_n in th.archi_string.split('-'):
    n = int(str_n)
    v1.append(m.mu.HyperConv3D(
      n, kernel_size=th.kernel_size, activation='relu'))
  v1.append(m.mu.HyperConv3D(1, 1))

  vertices = [v1,
              m.mu.HyperConv3D(1, kernel_size=5),
              m.mu.HyperConv3D(1, kernel_size=5),
              m.AdaptiveMerge(model.input_)]
  N = len(vertices)
  edges = ''
  for i in range(len(vertices) - 1): edges += '1' + '0' * i + ';'

  if th.include_input: edges += '1' * N
  else: edges += '0' + '1' * (N - 1)

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
  th.dose = ['1-4', '1-10', '1-20', '1-50', '1-100'][1]
  th.data_config = f'epsilon dataset=01-ULD dose={th.dose}'

  th.val_size = 1
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  th.data_shape = [1, 608, 440, 440, 1]
  th.buffer_size = 18

  th.norm_by_feature = True
  th.int_para_1 = 2
  th.suffix = ''
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

  th.archi_string = '4-4-4'
  th.kernel_size = 3
  th.normalize_energy = True
  th.include_input = True


  th.use_tanh = 0
  th.learn_delta = False
  th.rand_batch = True
  th.use_sigmoid = False

  th.ne_gamma = 0.00
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = 0
  # th.probe_cycle = th.updates_per_round

  th.batch_size = 4
  th.val_batch_size = 2

  th.loss_string = 'rmse'
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.001
  th.val_decimals = 7

  th.train = True

  # th.developer_code += 'chip'
  # th.developer_code += 'ecc'

  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, f'({th.archi_string}ks{th.kernel_size})')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
