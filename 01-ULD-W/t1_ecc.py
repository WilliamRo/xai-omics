import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'ecc'
id = 3
def model():
  th = core.th

  model = m.get_initial_model()

  # for str_c in th.archi_string.split('-'):
  #   c = int(str_c)
  #   model.add(m.mu.HyperConv3D(c, th.kernel_size, activation='relu'))

  fg = None
  if 'ecc' in th.developer_code: fg = m.gen_ecc_filter
  model.add(m.mu.HyperConv3D(1, th.kernel_size, filter_generator=fg))

  # model.add(m.mu.HyperConv3D(1, kernel_size=1))

  return m.finalize(model)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0

  # th.developer_code = 'self2self'
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.dose = ['1-2', '1-4', '1-10', '1-20', '1-50', '1-100'][4]
  th.data_config = fr'epsilon dataset=01-ULD dose={th.dose}'

  th.val_size = 1
  th.test_size = 1
  th.data_shape = [1, 500, 256, 256, 1]
  # th.data_shape = [1, 400, 400, 400, 1]

  th.norm_by_feature = True
  th.int_para_1 = 2
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

  th.archi_string = 'null'
  th.kernel_size = 3
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.early_stop = True
  th.patience = 5
  th.probe_cycle = th.updates_per_round

  th.batch_size = 1
  th.val_batch_size = 1

  th.loss_string = ['mae', 'rmse'][1]
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.001
  th.val_decimals = 7

  th.train = 1

  th.developer_code += 'chip'
  # th.developer_code += 'ecc'
  th.uld_batch_size = 10
  th.thickness = 20
  th.updates_per_round = 20

  if not th.train:
    th.visible_gpu_id = -1
    th.data_shape = [1, 656, 440, 440, 1]
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(
    model_name, f'({th.archi_string})ks{th.kernel_size}-{th.developer_code}')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
