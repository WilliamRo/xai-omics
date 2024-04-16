from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression

import aa_compare_auc_delong_xu as compare_auc_delong_xu
import matplotlib

matplotlib.use('TkAgg')  # 或者使用其他可用的后端，如 'QtAgg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb


# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBClassifier, plot_importance
# import random
# import seaborn as sns

def fix_data():
	data_img = pd.read_excel(r"../data/data.xlsx")
	data_img = data_img.select_dtypes(include=[np.number])
	# data_img = data_img.fillna(0)

	data_linc = pd.read_excel(r"../data/clinic.xlsx")
	data_img.insert(0, 'lab', data_linc['lab'])
	np.seterr(invalid='ignore')

	x = data_img[data_img.columns[1:]]
	y = data_img['lab']

	col_names = x.columns
	x = x.astype(np.float64)
	x = StandardScaler().fit_transform(x)
	x = pd.DataFrame(x)
	x.columns = col_names
	# print(x.head())

	df_img_merge = pd.concat([y, x], axis=1)
	df_img_1 = data_img.loc[data_img['lab'] == 1]
	df_img_0 = data_img[(data_img.lab == 0)]

	return df_img_1, df_img_0, df_img_merge


def fix_data_new(label_name='trg23'):
	data_img = pd.read_excel(r"../data/data.xlsx")
	data_img = data_img.select_dtypes(include=[np.number])
	# data_img = data_img.fillna(0)

	data_linc = pd.read_excel(r"../data/clinic.xlsx")
	data_img.insert(0, 'lab', data_linc[label_name])
	np.seterr(invalid='ignore')

	x = data_img[data_img.columns[1:]]
	y = data_img['lab']

	col_names = x.columns
	x = x.astype(np.float64)
	x = StandardScaler().fit_transform(x)
	x = pd.DataFrame(x)
	x.columns = col_names
	# print(x.head())

	df_img_merge = pd.concat([y, x], axis=1)
	df_img_1 = data_img.loc[data_img['lab'] == 1]
	df_img_0 = data_img[(data_img.lab == 0)]

	return df_img_1, df_img_0, df_img_merge


def t_test_feature(df_img_1, df_img_0, df_merge):
	feature_t = []
	for colNames in df_merge.columns[1:]:
		if 0.05 < stats.levene(df_img_0[colNames], df_img_1[colNames])[1]:
			t, p = stats.ttest_ind(df_img_0[colNames], df_img_1[colNames])
		else:
			t, p = stats.ttest_ind(df_img_0[colNames], df_img_1[colNames], equal_var=False)
		if p < 0.05:
			feature_t.append(colNames)
	print('t test %.f个' % len(feature_t), '\n')
	return feature_t


def t_test_feature2(df_img_1, df_img_0, df_merge):
	feature_t = []
	for colNames in df_merge.columns[1:]:
		if 0.05 < stats.levene(df_img_0[colNames], df_img_1[colNames])[1]:
			if stats.ttest_ind(df_img_0[colNames], df_img_1[colNames])[1] < 0.05:
				feature_t.append(colNames)
		else:
			if stats.ttest_ind(df_img_0[colNames], df_img_1[colNames], equal_var=False)[1] < 0.05:
				feature_t.append(colNames)
	print('2 t test %.f个' % len(feature_t), '\n')
	return feature_t


def vif_test_feature(data_feature):
	data_add = sm.add_constant(data_feature.iloc[:, 1:])  # 需要常数项列
	vif = pd.DataFrame()
	vif["变量"] = data_add.columns
	vif["VIF"] = [variance_inflation_factor(data_add.values, i) for i in range(data_add.shape[1])]
	print(vif)
	feature_vif = [x for x in vif.loc[vif["VIF"] < 10]['变量']]
	if 'const' in feature_vif:
		feature_vif.remove('const')
	print('vif test之后%.f个' % len(feature_vif))
	return feature_vif


def lasso_test_feature(df_img_1, df_img_0, feature_t):
	if 'lab' not in feature_t:
		feature_t = ['lab'] + feature_t
	df_feature_0 = df_img_0[feature_t]
	df_feature_1 = df_img_1[feature_t]

	df_feature = pd.concat([df_feature_0, df_feature_1])
	df_feature = shuffle(df_feature)
	original_indices = df_feature.index.tolist()

	df_feature.index = range(len(df_feature))

	x_lasso = df_feature[df_feature.columns[1:]]
	x_lasso = x_lasso.apply(pd.to_numeric, errors='ignore')
	y_lasso = df_feature['lab']

	lasso_col_names = x_lasso.columns
	x_lasso = x_lasso.astype(np.float64)
	x_lasso = StandardScaler().fit_transform(x_lasso)
	x_lasso = pd.DataFrame(x_lasso)
	x_lasso.columns = lasso_col_names

	x_lasso_drew = x_lasso
	y_lasso_drew = y_lasso
	print('x_lasso筛选————', x_lasso.shape)
	print('y_lasso筛选————', len(y_lasso))

	alphas_select = np.logspace(-3, 1, 50)
	model_lasso_cv = LassoCV(alphas=alphas_select, cv=2, max_iter=100000).fit(x_lasso, y_lasso)
	print('最佳alpha为————%.6f' % model_lasso_cv.alpha_)

	coef = pd.Series(model_lasso_cv.coef_, index=x_lasso.columns)  # 最佳alpha值下名称及系数
	print("lasso得到%.0f个————" % len(coef[coef != 0]))
	# print("lasso————" + str(sum(coef != 0)))  # 另一种
	index = coef[coef != 0].index
	x_lasso = x_lasso[index]
	print(x_lasso.shape)
	# print(coef[coef != 0])

	new_indices = df_feature.index.tolist()
	index_mapping = {new_index: original_indices[i] for i, new_index in enumerate(new_indices)}

	return x_lasso, y_lasso, x_lasso_drew, y_lasso_drew, index_mapping  # 返回lasso


def drew_lasso_(x, y):
	x_lasso_draw = x.values
	y_lasso_draw = y.values
	_, _, coefs = linear_model.lars_path(x_lasso_draw, y_lasso_draw, method='lasso', verbose=True)
	xx = np.sum(np.abs(coefs.T), axis=1)
	xx /= xx[-1]
	plt.rcParams['backend'] = 'SVG'
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.plot(xx, coefs.T)
	y_min, y_max = plt.ylim()
	plt.vlines(xx, y_min, y_max, linestyle='dashed')
	plt.style.use('ggplot')
	plt.xlabel('|coef| / max|coef|')
	plt.ylabel('Coefficients')
	plt.title('LASSO Path')
	plt.axis('tight')
	plt.rcParams['axes.unicode_minus'] = False
	plt.savefig('lasso1.svg', format='SVG')
	plt.show()  # lasso图


def logistic_test_backup(x, y):
	print('\n' * 5 + '-' * 20 + 'Logistic Regression' + '-' * 20 + '\n' * 5)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
	log = sm.Logit(y_train, x_train)
	result = log.fit()

	# print('lg', result.summary())
	# print('lg', np.exp(result.params))

	params = result.params
	conf = result.conf_int()
	conf['OR'] = params
	conf.columns = ['2.5%', '97.5%', 'OR']
	# print('lg', np.exp(conf))

	y_test_pre = result.predict(x_test)
	# print('预测', len(y_test_pre))

	y_test_pre = [x for x in y_test_pre]
	total, hit = 0, 0
	for i in range(len(y_test_pre)):
		if 0.5 < y_test_pre[i]:  # cut_off
			total += 1
			if y[i] == 1:
				hit += 1
			else:
				pass
		else:
			pass
	# print('%d, %d, %.2f' % (total, hit, 100.0 * hit / total))
	return y_test, y_test_pre


def logistic_test(x, y):
	print('\n' * 5 + '-' * 20 + 'Logistic Regression' + '-' * 20 + '\n' * 5)
	logistic_model = LogisticRegression()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
	log = logistic_model.fit(x_train, y_train)
	score = log.score(x_test, y_test)

	y_pred = logistic_model.predict(x_test)

	return y_test, y_pred


def xgb_test(x, y):
	print('\n' * 5 + '-' * 20 + 'XGB Regressor' + '-' * 20 + '\n' * 5)
	xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
	                          colsample_bytree=0.3,
	                          learning_rate=0.1,
	                          max_depth=5,
	                          n_estimators=10,
	                          alpha=10)
	data_matrix = xgb.DMatrix(x, y)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
	xg_reg.fit(x_train, y_train)
	# pre = xg_reg.predict(x_test)
	# print('test————', pre)
	# print(mean_squared_error(pre, y_test))

	params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3,
	          'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}
	cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3,
	                    num_boost_round=50, early_stopping_rounds=10,
	                    metrics="rmse", as_pandas=True, seed=123)
	# xgb.plot_importance(xg_reg)
	# plt.show()
	# print('xgb————', cv_results.head())

	# 预测值
	y_probably_xgb = xg_reg.predict(x_test)
	return y_test, y_probably_xgb  # xgb啊？TODO


def svm_test(x, y):
	print('\n' * 5 + '-' * 20 + 'SVM' + '-' * 20 + '\n' * 5)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
	model_svm = svm.SVC(kernel='rbf', gamma='auto', probability=True).fit(x_train, y_train)
	score_svm = model_svm.score(x_test, y_test)
	# print(f'SVM Accuracy: {score_svm}')

	# params opt，参数选择
	cs = np.logspace(-1, 3, 10, base=2)
	gammas = np.logspace(-4, 1, 20, base=2)
	param_grid = dict(C=cs, gamma=gammas)
	grid = GridSearchCV(svm.SVC(kernel='rbf', ), param_grid=param_grid, cv=3).fit(x, y)
	# print('最优————', grid.best_params_)
	c1 = grid.best_params_['C']
	gamma = grid.best_params_['gamma']

	# svm params，参数优化后
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
	model_svm = svm.SVC(kernel='rbf', C=c1, gamma=gamma, probability=True).fit(x_train, y_train)
	score_svm = model_svm.score(x_test, y_test)
	# print(f'SVM Accuracy: {score_svm}')

	# y预测值
	y_probably_svm = model_svm.predict_proba(x_test)
	y_score_svm = y_probably_svm[:, 1]
	# print(y_score_svm)
	return y_test, y_score_svm, model_svm  # SVM???TODO


def random_forest_test(x, y):
	print('\n' * 5 + '-' * 20 + 'Random Forest' + '-' * 20 + '\n' * 5)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

	param_test1 = {"n_estimators": range(1, 101, 3)}
	g_search1 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_test1,
	                         scoring='roc_auc', cv=10)
	g_search1.fit(x_train, y_train)
	# print(g_search1.grid_scores_)
	print(g_search1.cv_results_)
	print('rf————', g_search1.best_params_['n_estimators'])
	print("rf————%f" % g_search1.best_score_)
	n_estimators = g_search1.best_params_['n_estimators']

	param_test2 = {"max_features": range(1, len(x.columns), 2)}
	g_search2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=n_estimators,
	                                                          random_state=10),
	                         param_grid=param_test2, scoring='roc_auc', cv=3)
	g_search2.fit(x_train, y_train)
	# print(g_search2.grid_scores_)
	print(g_search2.cv_results_)
	print('rf————', g_search2.best_params_['max_features'])
	max_features = g_search2.best_params_['max_features']
	print('rf:%f' % g_search2.best_score_)

	rf_zy_model = RandomForestClassifier(n_estimators=n_estimators,
	                                     max_features=max_features,
	                                     oob_score=False, random_state=3)  # obbFalse
	rf_zy_model.fit(x_train, y_train)
	print(rf_zy_model.oob_score)
	print("rf%f" % rf_zy_model.oob_score)  # TODO
	score_rf_zy = rf_zy_model.score(x_test, y_test)
	print(f'RandomForest Accuracy: {score_rf_zy}')

	# 预测值
	y_test_pre = rf_zy_model.predict_proba(x_test)
	print('rf————', y_test_pre[:, 1])
	return y_test, y_test_pre[:, 1], rf_zy_model


def draw_roc(name, y, y_score, color):
	fpr, tpr, thresholds = roc_curve(y, y_score)
	roc_auc = auc(fpr, tpr)

	# 画 ROC 曲线
	plt.figure()
	plt.plot(fpr, tpr, color=color, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(f'Receiver Operating Characteristic (ROC) Curve --- {name}')
	plt.legend(loc="lower right")
	plt.show()


def draw_roc_all(input):
	fig, axes = plt.subplots(1, len(input), figsize=(15, 15))
	for i in range(len(input)):
		name, y, y_score, color = input[i]
		fpr, tpr, thresholds = roc_curve(y, y_score)
		roc_auc = auc(fpr, tpr)
		print(f'AUC: {roc_auc}')

		axes[i].plot(fpr, tpr, color=color, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
		axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		axes[i].set_xlabel('False Positive Rate')  # 使用 set_xlabel
		axes[i].set_ylabel('True Positive Rate')  # 使用 set_ylabel
		axes[i].set_title(f'ROC Curve --- {name}')  # 使用 set_title
	# axes[i].legend(loc="lower right")
	plt.show()


def drew_repeated_kf_(x, y, mod, n_splits=3, n_repeats=2):
	model = mod
	if isinstance(mod, RandomForestClassifier):
		title = 'ROC curve - Random Forest'
		print('\n' * 5 + '-' * 20 + 'Random Forest Cross Validation' + '-' * 20 + '\n' * 5)
	else:
		title = 'ROC curve - SVM'
		print('\n' * 5 + '-' * 20 + 'SVM Cross Validation' + '-' * 20 + '\n' * 5)

	repeated_kfo = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
	for train_index, test_index in repeated_kfo.split(x):
		x_train_pk = x.iloc[train_index]
		x_test_pk = x.iloc[test_index]
		y_train_pk = y.iloc[train_index]
		y_test_pk = y.iloc[test_index]

		model.fit(x_train_pk, y_train_pk)
		score = model.score(x_test_pk, y_test_pk)

		y_probably_pk = model.predict_proba(x_test_pk)  # 此处test
		fpr_pk, tpr_pk, thresholds_pk = roc_curve(
			y_test_pk, y_probably_pk[:, 1])
		auc_score_pk = auc(fpr_pk, tpr_pk)
		print(f'Score: {score}')
		print(f'AUC: {auc_score_pk}')

		plt.rcParams['backend'] = 'SVG'
		plt.rcParams['font.sans-serif'] = ['SimHei']
		plt.plot(fpr_pk, tpr_pk, label='ROC curve(area=%0.2f)' % auc_score_pk)
		plt.ylim(0, 1)
		plt.xlim(0, 1)
		plt.xlabel("fpr_")  # 假阳为横
		plt.ylabel('tpr_')
		plt.legend(loc="lower right")

	plt.title(title)
	plt.show()


def logisticRegression_cv(x, y, n_splits=3, n_repeats=2):
	print('\n' + '-' * 20 + 'Logistic Regression Cross Validation' + '-' * 20 + '\n')
	logistic_model = LogisticRegression()

	flag = True

	repeated_kfo = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
	for train_index, test_index in repeated_kfo.split(x):
		x_train_pk = x.iloc[train_index]
		x_test_pk = x.iloc[test_index]
		y_train_pk = y.iloc[train_index]
		y_test_pk = y.iloc[test_index]

		log = logistic_model.fit(x_train_pk, y_train_pk)

		# Calculate Accuracy
		score = log.score(x_test_pk, y_test_pk)

		# Calculate AUC
		y_probably_pk = log.predict_proba(x_test_pk)
		fpr_pk, tpr_pk, thresholds_pk = roc_curve(y_test_pk, y_probably_pk[:, 1])
		auc_score_pk = auc(fpr_pk, tpr_pk)

		# Calculate F1 Score
		y_pred_pk = log.predict(x_test_pk)
		f1_score_pk = f1_score(y_test_pk, y_pred_pk)

		# Calculate misclassified indice
		misclassified_indices = test_index[np.where(y_test_pk != y_pred_pk)[0]]

		print(f'AUC: {round(auc_score_pk, 2)}\t\t F1: {round(f1_score_pk, 2)}'
		      f'\t\t Accuracy: {round(score, 2)}'
		      f'\t\t Misclassified Indices: {misclassified_indices}')

		if auc_score_pk < 0.75: flag = False

	# 	plt.rcParams['backend'] = 'SVG'
	# 	plt.rcParams['font.sans-serif'] = ['SimHei']
	# 	plt.plot(fpr_pk, tpr_pk, label='ROC curve(area=%0.2f)' % auc_score_pk)
	# 	plt.ylim(0, 1)
	# 	plt.xlim(0, 1)
	# 	plt.xlabel("fpr_")  # 假阳为横
	# 	plt.ylabel('tpr_')
	# 	plt.legend(loc="lower right")
	#
	# plt.title('ROC curve - Logistic Regression')
	# plt.show()
	if flag: print('OK')
	else: print('---')

	return flag


def xgb_cv(x, y, n_splits=3, n_repeats=2):
	print('\n' + '-' * 20 + 'XGB Cross Validation' + '-' * 20 + '\n')
	xg_reg = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

	flag = True

	repeated_kfo = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

	for train_index, test_index in repeated_kfo.split(x):
		x_train_pk = x.iloc[train_index]
		x_test_pk = x.iloc[test_index]
		y_train_pk = y.iloc[train_index]
		y_test_pk = y.iloc[test_index]

		xg_reg.fit(x_train_pk, y_train_pk)

		# Calculate Accuracy
		score = xg_reg.score(x_test_pk, y_test_pk)

		# Calculate AUC
		y_probably_pk = xg_reg.predict_proba(x_test_pk)
		fpr_pk, tpr_pk, thresholds_pk = roc_curve(y_test_pk, y_probably_pk[:, 1])
		auc_score_pk = auc(fpr_pk, tpr_pk)

		# Calculate F1 Score
		y_pred_pk = xg_reg.predict(x_test_pk)
		f1_score_pk = f1_score(y_test_pk, y_pred_pk)

		# Calculate misclassified indice
		misclassified_indices = test_index[np.where(y_test_pk != y_pred_pk)[0]]

		print(f'AUC: {round(auc_score_pk, 2)}\t\t F1: {round(f1_score_pk, 2)}'
		      f'\t\t Accuracy: {round(score, 2)}'
		      f'\t\t Misclassified Indices: {misclassified_indices}')

		if auc_score_pk < 0.75: flag = False

	# 	plt.rcParams['backend'] = 'SVG'
	# 	plt.rcParams['font.sans-serif'] = ['SimHei']
	# 	plt.plot(fpr_pk, tpr_pk, label='ROC curve(area=%0.2f)' % auc_score_pk)
	#
	# plt.ylim(0, 1)
	# plt.xlim(0, 1)
	# plt.xlabel("fpr_")  # 假正率在 x 轴上
	# plt.ylabel('tpr_')  # 真正率在 y 轴上
	# plt.legend(loc="lower right")
	# plt.title('ROC curve - xgboost regression')
	# plt.show()
	if flag:
		print('OK')
	else:
		print('---')
	return flag


if __name__ == '__main__':
	label_name = ['trg23', 'trg3'][0]
	t_test_method = [t_test_feature, t_test_feature2][1]
	n_fold = 5
	n_repeat = 1

	df_img_pos1, df_img_neg0, df_merge1 = fix_data_new(label_name)
	# t_feature = t_test_method(df_img_pos1, df_img_neg0, df_merge1)
	# x_lasso1, y_lasso1, xd, yd, index_mapping = lasso_test_feature(
	# 	df_img_pos1, df_img_neg0, t_feature)
	# for v in x_lasso1.columns.values: print(v)

	feature_names = [
		'wavelet-LHL_firstorder_StandardDeviation',
		'wavelet-LHL_glrlm_GrayLevelVariance',
		'wavelet-HLL_gldm_GrayLevelVariance',
		'wavelet-HLL_glrlm_RunLengthNonUniformity',
		'wavelet-HHL_glcm_Idn',
		'original_shape_Compactness2.1',
		'original_glrlm_GrayLevelNonUniformityNormalized.1',
		'log-sigma-3-mm-3D_glcm_Imc2.1',
		'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
		'wavelet-HLL_ngtdm_Coarseness.1',
		'wavelet-HLH_glrlm_ShortRunEmphasis.1',
		'wavelet-HHL_glrlm_ShortRunEmphasis.1']

	x_lasso1 = df_merge1.filter(items=feature_names)
	y_lasso1 = df_merge1['lab']

	save_feature = 0
	if save_feature:
		import pickle
		feature_dict = {
			'x_lasso1': x_lasso1.__array__(),
			'y_lasso1': y_lasso1.__array__(),
			'feature_name': feature_names
		}
		with open(r'../data/selected_feature_dict.pkl', 'wb') as pickle_file:
			pickle.dump(feature_dict, pickle_file)
			print('Feature dict saved successfully!!!')
	#
	# color1 = ['#fff352', '#529c47', '#674196', '#ea5550']
	#
	# lr_y, lr_y_score = logistic_test(x_lasso1, y_lasso1)
	# xgb_y, xgb_y_score = xgb_test(x_lasso1, y_lasso1)
	# y_rf, y_rf_predict, random_f_m = random_forest_test(x_lasso1, y_lasso1)
	# y_svm, y_svm_pre, svm_m = svm_test(x_lasso1, y_lasso1)
	#
	# roc_drawing_input = [['lr', lr_y, lr_y_score, color1[0]],
	#                      ['xgb', xgb_y, xgb_y_score, color1[1]],
	#                      ['rf', y_rf, y_rf_predict, color1[2]],
	#                      ['svm', y_svm, y_svm_pre, color1[3]]]
	# draw_roc_all(roc_drawing_input)

	lr_flag = logisticRegression_cv(x_lasso1, y_lasso1, n_fold, n_repeat)
	xgb_flag = xgb_cv(x_lasso1, y_lasso1, n_fold, n_repeat)
# drew_repeated_kf_(x_lasso1, y_lasso1, random_f_m, n_fold, n_repeat)
# drew_repeated_kf_(x_lasso1, y_lasso1, svm_m, n_fold, n_repeat)
