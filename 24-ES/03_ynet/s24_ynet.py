import sys, os
path = os.path.abspath(__file__)
for _ in range(3):
  path = os.path.dirname(path)
  sys.path.append(path)

from tframe.utils.script_helper import Helper
s = Helper()

from es_core import th
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
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 10000)
s.register('patience', 20)
updates_per_round = 10
s.register('updates_per_round', updates_per_round)
s.register('probe_cycle', updates_per_round)

# Data
s.register('random_flip', s.true_and_false)
s.register('random_rotation', s.true_and_false)
s.register('random_noise', s.true_and_false)
s.register('random_crop', True)

# Model
s.register('model_num', 3)
s.register('archi_string',
           '8-5-2-1-relu-mp', '8-5-2-3-relu-mp')
# s.register('archi_string',
#            '8-3-2-1-relu-mp', '4-3-2-2-relu-mp')
s.register('alpha_total', 1.0)
s.register('alpha_lesion', 0.6, 1.0)
s.register('alpha_region', 0.1, 0.9)
s.register('bias_initializer', 20)
s.register('if_pre_train', False)

# Optimizer
s.register('lr', 0.001, 0.0001)
s.register('batch_size', 10, 32)
s.register('batchlet_size', 8)

# s.configure_engine(times=5)
s.configure_engine(strategy='skopt', criterion='Best dice_accuracy',
                   greater_is_better=True, add_script_suffix=True)
s.run(rehearsal=False)
