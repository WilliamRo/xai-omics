from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from roma import console



# ------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------
data_path = r'D:/data/BAMIA/BAMIA-All-851.omix'

omix = Omix.load(data_path)
omix_1, omix_2 = omix.split(1, 1)

assert isinstance(omix_1, Omix)
assert isinstance(omix_2, Omix)

# omix_1.show_in_explorer()
# exit()

# omix_1.report()
# omix_2.report()
# ------------------------------------------------------------------------
# Fit pipeline
# ------------------------------------------------------------------------
pi = Pipeline(omix_1, ignore_warnings=1, save_models=1)
M = 1
pi.create_sub_space('lasso', repeats=M, show_progress=1)
N = 5
pi.fit_traverse_spaces('lr', repeats=N, nested=1, show_progress=1, verbose=1)
pi.report()

# ------------------------------------------------------------------------
# Evaluate pipeline
# ------------------------------------------------------------------------
console.section('Omix 1')
pkg = pi.evaluate_best_pipeline(omix_1, rank=1)
pkg.report()

console.section('Omix 2')
pkg = pi.evaluate_best_pipeline(omix_2, rank=1)
pkg.report()


