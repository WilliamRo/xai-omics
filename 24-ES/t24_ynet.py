import es_core as core
import es_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'ynet'
id = 3
def model():
  from es_core import th
  if th.model_num == 2:
    return m.get_ynet_2(th.archi_string)
  elif th.model_num == 3:
    return m.get_ynet_3(th.archi_string)


def main(_):
  console.start('{} on Esophagus Segmentation'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0

  th.allow_growth = False
  th.gpu_memory_fraction = 0.7
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.ratio_of_dataset = '80:10:18'
  th.random_flip = 0
  th.random_rotation = 0
  th.random_noise = 0

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

  th.archi_string = '4-3-2-1-relu-mp'
  th.if_pre_train = False
  th.model_num = 3

  th.alpha_total = 1.0
  th.alpha_region = 1.0
  th.alpha_lesion = 1.0
  th.bias_initializer = 99

  th.tic_toc = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 5
  th.early_stop = True
  th.updates_per_round = 5
  th.probe_cycle = th.updates_per_round
  th.patience = 10

  th.batch_size = 4
  th.batchlet_size = 2

  th.val_batch_size = 2
  th.eval_batch_size = 2

  th.optimizer = 'adam'
  th.learning_rate = 0.0002
  th.learning_rate = 0.2

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
