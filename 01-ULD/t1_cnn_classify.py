import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn'
id = 4
def model():
  from tframe.layers.pooling import ReduceMean
  mu = m.mu
  th = core.th

  model = m.get_initial_model()
  # model.add(ReduceMean(axis=[1, 2, 3]))

  for i, c in enumerate(th.archi_string.split('-')):
    if c == 'p':
      model.add(mu.MaxPool3D(pool_size=2, strides=3))
    else:
      c = int(c)
      model.add(mu.HyperConv3D(filters=c, kernel_size=th.kernel_size,
                               activation=th.activation))
  # model.add(ReduceMean(axis=[1, 2, 3]))
  model.add(mu.Flatten())

  return m.finalize(model)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = fr'explore dataset=01-ULD'

  th.val_size = 64
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128

  th.data_shape = [1, 32, 160, 160, 1]
  th.norm_by_feature = True
  th.classify = True
  if th.classify:
    th.input_shape = th.data_shape[1:]
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  # th.suffix += f'_w{th.window_size}_s{th.slice_size}'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '16-p-32-p-32-p-64-p-64'
  th.kernel_size = 5
  th.activation = 'lrelu'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = 0
  # th.updates_per_round

  th.batch_size = 64
  th.val_batch_size = 32

  th.buffer_size = 18

  th.opt_str = 'adam'
  th.optimizer = th.opt_str
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  th.save_model_at_the_end = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)

  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

