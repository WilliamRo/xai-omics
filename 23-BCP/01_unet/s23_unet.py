import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from bcp_core import th
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
s.register('gpu_memory_fraction', 0.9)
s.register('rehearse', False)
s.register('tic_toc', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 5000)
s.register('patience', 15)

# Data
s.register('random_noise', s.true_and_false)
s.register('random_flip', s.true_and_false)
s.register('random_rotation', s.true_and_false)

# Model
s.register('filter', 8, 16, hp_type=list)
s.register('kernel_size', 5, 7, hp_type=list)
s.register('depth', 2, 3, hp_type=list)
s.register('width', 1, 2, hp_type=list)
s.register('activation', 'relu', 'lrelu')

# Optimization
s.register('lr', 0.01, 0.001, 0.0001, 0.00001)
s.register('batch_size', 10, 24)

# s.configure_engine(times=5)
s.configure_engine(strategy='skopt', criterion='Best dice_accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)


