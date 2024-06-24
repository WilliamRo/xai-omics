from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline
from pictor.xomics.ml.dr.dr_engine import DREngine
from pictor.xomics.ml.ml_engine import FitPackage



data_path = r'D:/data/BAMIA/BAMIA-All-851.omix'

omix = Omix.load(data_path)
omix_1, omix_2 = omix.split(1, 1)

assert isinstance(omix_1, Omix)
assert isinstance(omix_2, Omix)

omix_1.show_in_explorer()
# omix_1.report()
# omix_2.report()
#
# pi = Pipeline(omix_1, ignore_warnings=1, save_models=1)
# M = 1
# pi.create_sub_space('lasso', repeats=M, show_progress=1)
# N = 5
# pi.fit_traverse_spaces('lr', repeats=N, show_progress=1)
# pi.report()
#
#
# ranking = pi.pipeline_ranking
# print('AUC:', [r[0] for r in ranking])
#
# best_dr: DREngine = ranking[0][1]
# best_pkg: FitPackage = ranking[0][2]
#
# ext_omix_reduced = best_dr.reduce_dimension(omix_2)
# pkg = best_pkg.evaluate(ext_omix_reduced)
# pkg.report()
#
# train_omix_reduced = best_dr.reduce_dimension(omix_1)
# pkg = best_pkg.evaluate(train_omix_reduced)
# pkg.report()


