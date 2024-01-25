import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from fc_core import th
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
s.register('epoch', 100)
s.register('patience', 20)
# s.register('archi_string', '32-16', '64-32', '64-16')
s.register('archi_string', '32-16')

# Data
k_fold = 5
s.register('cross_validation', True)
s.register('k_fold', k_fold)
s.register('val_fold_index', list(range(k_fold)))
# s.register('val_fold_index', k_fold - 1)

# Optimization
# s.register('lr', 0.01, 0.001, 0.0001, 0.00001)
# s.register('batch_size', 16, 64)
s.register('lr', 0.00140)
s.register('batch_size', 64)

s.configure_engine(times=15, add_script_suffix=True)
# s.configure_engine(strategy='skopt', criterion='Best F1',
#                    greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
