import numpy as np

import uld_core as core
import uld_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'unet'
id = 1
def model():
  th = core.th

  return m.get_unet(th.archi_string)


def main(_):
  console.start('{} on Ultra Low Dose task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = r'delta dataset=01-ULD dose=1-4'

  th.val_size = 30
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128

  th.slice_num = 608
  th.use_tanh = 0
  th.use_color = False
  th.use_suv = False
  th.norm_by_feature = False
  th.train_self = not th.norm_by_feature
  # th.use_clip = 1.0

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  if th.train_self:
    th.suffix += '_self'
  if th.use_suv:
    th.suffix += '_SUV'
  if th.norm_by_feature:
    th.suffix += '_normBF'
  if th.use_tanh != 0:
    th.suffix += f'_tanh{th.use_tanh}'
  if th.use_clip != np.Inf:
    th.suffix += f'_clip{th.use_clip}'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '4-3-3-2-lrelu'
  # th.archi_string = '16-5-2-3-relu-mp'
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
  th.optimizer = 'adam'
  # th.optimizer = 'sgd'
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
