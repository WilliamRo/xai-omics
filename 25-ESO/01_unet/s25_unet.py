import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from eso_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure datas set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
s.register('gpu_memory_fraction', 0.8)
s.register('rehearse', False)
s.register('tic_toc', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 5000)
s.register('patience', 20)

# Data
s.register('random_flip', s.true_and_false)
s.register('random_rotation', s.true_and_false)
s.register('random_noise', s.true_and_false)
s.register('random_crop', True)

# Model
s.register('filter', 4, 8, hp_type=list)
s.register('kernel_size', 3, 5, 7, hp_type=list)
s.register('depth', 3, 4, 5, hp_type=list)
s.register('width', 1, 2, hp_type=list)
s.register('activation', 'relu', 'lrelu')

# Optimization
s.register('lr', 0.01, 0.001, 0.0001, 0.00001)
s.register('batch_size', 8, 32)
s.register('batchlet_size', 10)

# s.configure_engine(times=5)
s.configure_engine(strategy='skopt', criterion='Best dice_accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
