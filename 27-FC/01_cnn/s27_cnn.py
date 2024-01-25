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
s.register('epoch', 5000)
s.register('patience', 40)

# Data
s.register('label_type', 'trg23')

# Model
s.register('kernel_size', 100, 300)
s.register('activation', 'relu', 'lrelu')
s.register('archi_string',
           '64-p-32-p-16', '32-p-16-p-8')

# Optimization
s.register('lr', 0.01, 0.001, 0.0001, 0.00001)
s.register('batch_size', 32, 64)

# s.configure_engine(times=5)
s.configure_engine(strategy='skopt', criterion='Best Accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
