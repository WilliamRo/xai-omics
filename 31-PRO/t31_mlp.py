import pro_core as core
import pro_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
id = 2
def model():
  th = core.th

  return m.get_mlp(th.archi_string)


def main(_):
  console.start('{} on Features Classification in Prostate'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------

  th.ratio_of_dataset = '7:1:2'

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
  th.kernel_size = 5
  th.activation = 'relu'
  # th.loss_string = 'weighted_cross_entropy'
  th.loss_string = 'cross_entropy'

  th.archi_string = '4-4'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = True
  th.probe_cycle = th.updates_per_round // 2
  th.patience = 20

  th.batch_size = 64

  th.val_batch_size = 2
  th.eval_batch_size = 2

  th.optimizer = 'adam'
  th.learning_rate = 0.001
  # th.learning_rate = 0.003
  th.train = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  # th.mark = '{}({})'.format(
  #   model_name, th.archi_string + '-' + th.link_indices_str)
  th.mark = '(31-PRO)_{}({})'.format(
    model_name, th.archi_string)
  # th.mark += th.data_config.replace('>', '-')
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
