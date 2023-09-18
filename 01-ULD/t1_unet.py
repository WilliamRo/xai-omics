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
  th.dose = '1-4'
  th.data_config = fr'epsilon dataset=01-ULD dose={th.dose}'

  th.val_size = 30
  th.test_size = 1

  th.window_size = 128
  th.slice_size = 128
  # th.eval_window_size = 128
  th.data_shape = [1, 656, 440, 440, 1]


  # th.use_suv = False
  th.norm_by_feature = True
  # th.output_result = True
  # th.train_self = not th.norm_by_feature
  # th.max_clip = 1.0

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

  th.archi_string = '4-3-3-2-lrelu'
  # th.archi_string = '8-5-2-3-lrelu'

  th.use_tanh = 0
  th.learn_delta = False
  th.rand_batch = True
  th.use_sigmoid = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.patience = 15
  th.probe_cycle = th.updates_per_round

  th.batch_size = 4
  # th.batchlet_size = 2
  th.val_batch_size = 4

  th.buffer_size = 18

  th.loss_string = 'rmse'
  th.opt_str = 'adam'

  th.optimizer = th.opt_str
  th.learning_rate = 0.0003
  th.val_decimals = 7

  th.clip_threshold = 5
  th.clip_method = 'value'

  th.train = True
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  if th.learn_delta:
    th.suffix += '_delta'
  if th.use_sigmoid:
    th.suffix += '_sig'
  if not th.rand_batch:
    th.suffix += '_fakeRandBS'
  if th.train_self:
    th.suffix += '_self'
  if th.use_suv:
    th.suffix += '_SUV'
  if not th.norm_by_feature:
    th.suffix += '_normSelf'
  if th.use_tanh != 0:
    th.suffix += f'_tanh{th.use_tanh}'
  if th.max_clip != None:
    th.suffix += f'_clip(0,{th.use_clip})'
  th.suffix += f'_lr{th.learning_rate:3e}_loss({th.loss_string})_BS{th.batch_size}'
  th.suffix += f'_{th.opt_str}'
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
