import bcp_core as core
import bcp_mu as m

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

  return m.get_unet_3d(th.archi_string, link_indices=th.link_indices)


def main(_):
  console.start('{} on Brain Analysis'.format(model_name.upper()))

  th = core.th
  th.rehearse = 1
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.ratio_of_dataset = '6:1:3'

  th.random_noise = True
  th.random_flip = True
  th.random_rotation = True
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

  th.filter = 16
  th.kernel_size = 7
  th.depth = 3
  th.width = 2
  th.activation = 'relu'

  th.archi_string = '{}-{}-{}-{}-{}-mp'.format(
    th.filter, th.kernel_size, th.depth, th.width, th.activation)
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.tic_toc = True
  th.epoch = 2000
  th.early_stop = True
  th.probe_cycle = th.updates_per_round // 2
  th.patience = 10

  th.batch_size = 16
  # th.batchlet_size = 2

  th.val_batch_size = 2
  th.eval_batch_size = 2

  th.optimizer = 'adam'
  th.learning_rate = 0.003

  th.train = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '(23-BCP)_{}({})'.format(
    model_name, th.archi_string)
  # th.mark += th.data_config.replace('>', '-')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
