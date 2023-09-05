import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'conv'
id = 2
def model():
  mu = m.mu
  th = core.th

  model = m.get_initial_model()

  for str_c in th.archi_string.split('-'):
    c = int(str_c)
    model.add(mu.HyperConv3D(c, th.kernel_size, activation=th.activation))

  return m.finalize(model)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = r'explore dataset=01-ULD dose=1-20'

  th.val_size = 30
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128

  th.data_shape = [1, 608, 440, 440, 1]
  th.norm_by_feature = True

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.suffix += f'_{th.data_kwargs["dose"]}_w{th.window_size}_s{th.slice_size}'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '8-8'
  th.kernel_size = 5
  th.activation = 'lrelu'

  th.learn_delta = 0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = th.updates_per_round

  th.batch_size = 1
  th.val_batch_size = 1

  th.buffer_size = 18

  th.loss_string = 'rmse'
  th.opt_str = 'adam'
  th.optimizer = th.opt_str
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  if th.use_tanh != 0: th.mark += f'-tanh({th.use_tanh})'

  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
