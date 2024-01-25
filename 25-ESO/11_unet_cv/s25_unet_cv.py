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
s.register('gpu_memory_fraction', 0.85)
s.register('rehearse', False)
s.register('tic_toc', True)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 5000)
s.register('patience', 20)

# Cross Validation
k_fold = 5
s.register('cross_validation', True)
s.register('k_fold', k_fold)
s.register('val_fold_index', list(range(k_fold)))

# Data
s.register('random_flip', True)
s.register('random_rotation', True)
s.register('random_noise', False)
s.register('random_crop', True)

# Model
s.register('filter', 4)
s.register('kernel_size', 5)
s.register('depth', 4)
s.register('width', 1)
s.register('activation', 'relu')

# Optimization
s.register('lr', 0.0001674389146321581)
s.register('batch_size', 22)
s.register('batchlet_size', 10)

s.configure_engine(times=15, add_script_suffix=True)
# s.configure_engine(strategy='skopt', criterion='Best dice_accuracy',
#                    greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
