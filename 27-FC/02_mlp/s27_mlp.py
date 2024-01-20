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
s.register('patience', 30)

# Data
s.register('label_type', 'trg_01_23')

# Model
s.register('activation', 'relu')
s.register('archi_string','32-16')

# Optimization
s.register('lr', 0.001, 0.0001, 0.00001)
s.register('batch_size', 32, 96)

# s.configure_engine(times=10)
s.configure_engine(strategy='skopt', criterion='Best F1', greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
