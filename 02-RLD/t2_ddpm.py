
import rld_core as core
import rld_mu as m
import rld_tu as tu

from tframe import console, SaveMode
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir



# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'ddpm'
id = 10
def model():
  th = core.th
  model = m.get_ddpm_container(time_steps=th.time_step, time_dim=th.time_dim)
  unet = m.TimeUNet(th.archi_string, model, th.dimension)
  unet.add_to(model)

  return m.finalize(model)


def main(_):
  console.start('{} on PET/CT reconstruct task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.visible_gpu_id = 0
  th.data_config = fr'alpha dataset=02-RLD'

  th.val_size = 4
  th.test_size = 5

  th.windows_size = None
  # th.eval_windows_size = [1, 128, 128]
  th.data_shape = [560, 440, 440]

  th.dimension = 2
  if th.dimension == 2:
    th.input_shape = th.input_shape[1:]

  th.gan = False
  th.probe_func = tu.ddpm_probe
  th.probe_per_round = 0.1

  th.data_set = ['30G', '240G']
  th.process_param = {
    'ct_window': None,
    'norm': 'PET',  # only min-max,
    'shape': th.data_shape[::-1],  # [320, 320, 240]
    'crop': [10, 0, 0][::-1],  # [30, 30, 10]
    'clip': None,  # [1, None]
  }

  th.noCT = True
  if th.noCT:
    th.input_shape[-1] = 1
  # th.use_suv = False

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''
  th.suffix += f'_win{tuple(th.windows_size)}' if th.windows_size else ''


  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.archi_string = '64-3-3-2-lrelu'

  th.ddpm = True
  th.time_step = 1000
  th.time_dim = 512

  th.use_sigmoid = False
  th.clip_off = False
  th.output_conv = True
  # th.use_res = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.early_stop = False
  th.patience = 15
  th.validation_per_round = 0

  th.batch_size = 64
  th.batchlet_size = 32
  th.val_batch_size = 2

  th.buffer_size = 16

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
  th.suffix += '_noCT' if th.noCT else ''
  th.suffix += f'_{th.dimension}D_{th.data_set[0]}to{th.data_set[1]}'
  th.suffix += f'_lr{th.learning_rate}_bs{th.batch_size}_{th.opt_str}'
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
