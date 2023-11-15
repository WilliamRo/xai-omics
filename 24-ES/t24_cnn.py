import es_core as core
import es_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn'
id = 2
def model():
  th = core.th

  return m.get_cnn()


def main(_):
  console.start('{} on Esophagus Segmentation'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.ratio_of_dataset = '8:2'
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

  th.archi_string = '2-5-2-2-relu-mp'
  # th.archi_string = '8-3-2-3-relu-mp'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 2000
  th.early_stop = True
  th.probe_cycle = th.updates_per_round // 2
  th.patience = 10

  th.batch_size = 4
  th.batchlet_size = 4
  # th.gradlet_in_device = 1

  th.val_batch_size = 2
  th.eval_batch_size = 2

  th.optimizer = 'adam'
  # th.optimizer = 'sgd'
  th.learning_rate = 0.002

  th.train = True
  th.overwrite = 1

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
