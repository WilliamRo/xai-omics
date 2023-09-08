import mic_core as core
import mic_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'fcn'
id = 2
def model():
  th = core.th
  model_type, model_num = th.archi_string.split('-')

  if model_type == 'fcn' and model_num == '3':
    return m.get_fcn_3d_03()
  elif model_type == 'fcn' and model_num == '4':
    return m.get_fcn_3d_04()
  else:
    assert TypeError('No model!!!')


def main(_):
  console.start('{} on Medical Image Classification task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.ratio_of_dataset = '7:2:1'
  th.random_flip = 1
  th.random_rotation = 1
  th.random_noise = 1
  th.random_translation = 1

  th.cross_validation = False
  th.num_fold = 5

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = 'fcn-4'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 2000
  th.early_stop = True
  th.probe_cycle = th.updates_per_round // 2
  th.patience = 10

  th.batch_size = 128
  th.val_batch_size = 16
  th.batchlet_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.005

  th.train = True
  th.overwrite = True


  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  # th.mark = '{}({})'.format(
  #   model_name, th.archi_string + '-' + th.link_indices_str)
  th.mark = '{}({})'.format(
    model_name, th.archi_string)
  # th.mark += th.data_config.replace('>', '-')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
