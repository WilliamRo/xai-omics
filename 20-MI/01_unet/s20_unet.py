import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from mi_core import th
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
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 2000)
s.register('patience', 10)

s.register('lr', 0.01, 0.001, 0.0001)
s.register('batch_size', 4, 8)
s.register('th.archi_string', '2-3-1-2-lrelu', '2-3-2-2-lrelu')
s.register('th.random_flip', s.true_and_false)
s.register('th.random_rotation', s.true_and_false)
s.register('th.random_noise', s.true_and_false)

s.configure_engine(times=5)
s.run(rehearsal=False)
