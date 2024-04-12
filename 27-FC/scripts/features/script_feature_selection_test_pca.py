from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import time
import openpyxl
import os



def get_data(data_path, data_linc_path, label_name='trg23'):
	data_img = pd.read_excel(data_path)
	pids = data_img['PIDS']
	data_img = data_img.select_dtypes(include=[np.number])
	data_img.insert(0, 'pids', pids)
	# data_img = data_img.fillna(0)

	data_linc = pd.read_excel(data_linc_path)
	data_img.insert(0, 'lab', data_linc[label_name])
	np.seterr(invalid='ignore')

	exclude_col = ['lab', 'pids']
	y = data_img[exclude_col]
	x = data_img[data_img.columns[len(exclude_col):]]

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
	for colNames in df_merge.columns[2:]:
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
	for colNames in df_merge.columns[2:]:
		if 0.05 < stats.levene(df_img_0[colNames], df_img_1[colNames])[1]:
			if stats.ttest_ind(df_img_0[colNames], df_img_1[colNames])[1] < 0.05:
				feature_t.append(colNames)
	else:
		if stats.ttest_ind(df_img_0[colNames], df_img_1[colNames], equal_var=False)[1] < 0.05:
			feature_t.append(colNames)
	print('2 t test %.f个' % len(feature_t), '\n')
	return feature_t


def lasso_test_feature(df_img_1, df_img_0, feature_t):
	if 'lab' not in feature_t:
			feature_t = ['lab'] + feature_t
	df_feature_0 = df_img_0[feature_t]
	df_feature_1 = df_img_1[feature_t]

	df_feature = pd.concat([df_feature_0, df_feature_1])
	df_feature = shuffle(df_feature)
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
	return x_lasso, y_lasso, x_lasso_drew, y_lasso_drew  # 返回lasso


def feature_selection(raw_x, raw_y, ):
	pass


def calculate_macro_f1(y_true, y_pred):
	lb = LabelBinarizer()
	y_true_bin = lb.fit_transform(y_true)
	y_pred_bin = lb.transform(y_pred)

	f1 = f1_score(y_true_bin, y_pred_bin, average=None)

	macro_f1 = sum(f1) / len(f1)

	return macro_f1


def calculate_revised_f1(y_true, y_probability, threshold=0.15):
	revised_gt, revised_pred = get_revised_true_and_pred(
		y_true, y_probability, threshold)

	if len(revised_gt) == 0:
		return 0.0
	else:
		return calculate_macro_f1(revised_gt, revised_pred)


def roc_auc_ci(y_true, y_score, positive=1):
	AUC = roc_auc_score(y_true, y_score)
	N1 = sum(y_true == positive)
	N2 = sum(y_true != positive)
	Q1 = AUC / (2 - AUC)
	Q2 = 2*AUC**2 / (1 + AUC)
	SE_AUC = np.sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
	lower = AUC - 1.96*SE_AUC
	upper = AUC + 1.96*SE_AUC
	if lower < 0:
		lower = 0
	if upper > 1:
		upper = 1
	return lower, upper


def calculate_accuracy(y_true, y_pred):
	if not isinstance(y_true, np.ndarray):
		y_true = y_true.values
	if not isinstance(y_pred, np.ndarray):
		y_pred = y_pred.values.reshape(-1)

	assert y_true.shape == y_pred.shape

	return np.sum(y_true == y_pred) / len(y_pred)


def calculate_revised_accuracy(y_true, y_probability, threshold=0.15):
	revised_gt, revised_pred = get_revised_true_and_pred(
		y_true, y_probability, threshold)

	if len(revised_gt) == 0:
		return 0.0
	else:
		return np.sum(revised_pred == revised_gt) / len(revised_pred)


def get_revised_true_and_pred(y_true, y_probability, threshold=0.15):
	if not isinstance(y_true, np.ndarray):
		y_true = y_true.values
	if not isinstance(y_probability, np.ndarray):
		y_probability = y_probability.values

	assert len(y_true) == len(y_probability)

	revised_list = []
	for conf in y_probability:
		if np.abs(conf[0] - conf[1]) <= threshold * 2: revised_list.append(0)
		else: revised_list.append(1)

	revised_list = np.array(revised_list)

	revised_pred = y_probability[revised_list == 1][:, 1]
	revised_pred[revised_pred < threshold + 0.5] = 0
	revised_pred[revised_pred >= threshold + 0.5] = 1

	revised_gt = y_true[revised_list == 1]

	return revised_gt, revised_pred


def draw_confidence_2(y_true, y_probability, threshold_setting=None):
	if threshold_setting is None:
		threshold_setting = [0.5, 0.95, 0.05]

	count = round((threshold_setting[1] - threshold_setting[0]) / threshold_setting[2])
	threshold_list = [threshold_setting[2] * i for i in range(count + 1)]

	acc_list = []
	for thres in threshold_list:
		acc_list.append(calculate_revised_accuracy(y_true, y_probability, thres))

	acc_list = np.array(acc_list)
	threshold_list = np.array([t + threshold_setting[0] for t in threshold_list])

	plt.hist(threshold_list, weights=acc_list, bins=20, alpha=0.4, label='Confidence')
	plt.plot(threshold_list, acc_list, label='Confidence')
	plt.plot([0.5, 1.0], [0.5, 1.0], linestyle='--', lw=2, color='r', label='Diag', alpha=.8)

	plt.rcParams['backend'] = 'TkAgg'
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.ylim(0.5, 1.0)
	plt.xlim(0.5, 1.0)
	plt.xlabel("Threshold")
	plt.ylabel('Accuracy')
	plt.legend(loc="lower right")
	plt.title(f'Confidence')
	# plt.show()

	plt.clf()
	return acc_list, threshold_list


def draw_boxplot(data_list: list, data_name: list, title):
	assert len(data_list) == len(data_name)
	colors = [
		'63B2EE', '76DA91', 'F8CB7F', 'F89588', '7CD6CF',
		'9192AB', '7898E1', 'EFA666', 'EDDD86', '9987CE']

	box = plt.boxplot(data_list, patch_artist=True, showmeans=True, showfliers=True)
	plt.grid(True, axis='y')
	plt.xticks(list(range(1, len(data_list) + 1)), data_name)
	plt.title(title)
	plt.xlabel('Threshold')
	plt.ylabel('Accuracy')

	# Fill Colors
	for patch, color in zip(box['boxes'], colors):
		patch.set_facecolor(mcolors.to_rgba('#' + color))

	plt.show()


def traditional_method_cv(x, y, n_splits=1, auc_threshold=0.75, model_type='lr'):
	if model_type == 'lr':
		print('\n' + '-' * 20 + 'Logistic Regression Cross Validation' + '-' * 20 + '\n')
		model = LogisticRegression()
	elif model_type == 'xgb':
		print('\n' + '-' * 20 + 'XGB Cross Validation' + '-' * 20 + '\n')
		model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
	elif model_type == 'svm':
		print('\n' + '-' * 20 + 'SVM Cross Validation' + '-' * 20 + '\n')
		model = SVC(probability=True)
	elif model_type == 'rf':
		print('\n' + '-' * 20 + 'Random Forest Cross Validation' + '-' * 20 + '\n')
		model = RandomForestClassifier(n_estimators=100)
	else:
		assert TypeError('False Model Type !')

	# Parameter Setting and Initialization
	flag = True
	whole_misclassified_indices = []
	whole_pred, whole_truth, whole_probability = [], [], []
	mean_fpr = np.linspace(0, 1, 100)
	tprs, aucs = [], []
	colors = ['#A1A9D0', '#F0988C', '#B883D4', '#9E9E9E',
	          '#CFEAF1', '#C4A5DE', '#F6CAE5', '#96CCCB']

	repeated_kfo = RepeatedKFold(n_splits=n_splits, n_repeats=1)
	for fold_i, (train_index, test_index) in enumerate(repeated_kfo.split(x)):
		x_train = x.iloc[train_index]
		x_test = x.iloc[test_index]
		y_train = y.iloc[train_index]
		y_test = y.iloc[test_index]

		log = model.fit(x_train, y_train)
		y_probability = log.predict_proba(x_test)
		y_pred = log.predict(x_test)

		whole_pred.append(pd.DataFrame(y_pred, index=y_test.index))
		whole_truth.append(y_test)
		whole_probability.append(pd.DataFrame(y_probability, index=y_test.index))

		# Calculate Accuracy and Revised Accuracy
		acc = log.score(x_test, y_test)
		revised_acc = calculate_revised_accuracy(y_test, y_probability)

		# Calculate F1 Score
		f1 = calculate_macro_f1(y_test, y_pred)
		revised_f1 = calculate_revised_f1(y_test, y_probability)

		# Calculate AUC
		fpr, tpr, thresholds_pk = roc_curve(y_test, y_probability[:, 1])
		auc_score = auc(fpr, tpr)
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		aucs.append(auc_score)

		plt.plot(
			fpr, tpr, color=colors[fold_i], alpha=1, lw=2,
			label=f'Validation Set {fold_i + 1} (AUC = %0.2f)' % auc_score)

		# Calculate misclassified indice
		misclassified_indices = test_index[np.where(y_test != y_pred)[0]]
		whole_misclassified_indices.extend(misclassified_indices)

		# Calculate confusion matrix
		confusion_mtx = confusion_matrix(y_test, y_pred)

		# Print
		print(f'[Fold{fold_i + 1}]'
		      f'\t\t AUC: {auc_score: .4f}'
		      f'\t\t Macro F1: {f1: .4f}'
		      f'\t\t Revised Macro F1: {revised_f1: .4f}'
		      f'\t\t Acc: {acc: .4f}'
		      f'\t\t Revised Acc: {revised_acc: .4f}')

		if auc_score < auc_threshold: flag = False

	whole_pred = pd.concat(whole_pred).sort_index()
	whole_truth = pd.concat(whole_truth).sort_index()
	whole_probability = pd.concat(whole_probability).sort_index()

	# Calculate F1 Score and Revised F1 Score in total
	f1_all = calculate_macro_f1(whole_truth, whole_pred)
	revised_f1_all = calculate_revised_f1(whole_truth, whole_probability)

	# Calculate Accuracy and Revised Accuracy in total
	acc_all = calculate_accuracy(whole_truth, whole_pred)
	revised_acc_all = calculate_revised_accuracy(whole_truth, whole_probability)

	# Calculate Mean AUC
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(
		mean_fpr, mean_tpr, color=colors[-3],
		label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
		lw=4, alpha=1)
	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[-2], alpha=0.5)

	# Print
	print(f'[##SUM]'
	      f'\t\t AUC: {mean_auc: .4f}'
	      f'\t\t Macro F1: {f1_all: .4f}'
	      f'\t\t Revised Macro F1: {revised_f1_all: .4f}'
	      f'\t\t Acc: {acc_all: .4f}'
	      f'\t\t Revised Acc: {revised_acc_all: .4f}')

	# Uncertainty / total
	_, revised_prob = get_revised_true_and_pred(whole_truth, whole_probability)
	length_uncertainty = len(whole_probability) - len(revised_prob)
	length_total = len(whole_probability)

	print(f'Uncertainty / Total : {length_uncertainty} / {length_total}'
	      f'\t\t Ratio: {(length_uncertainty / length_total): .4f}')

	# Judge flag
	if flag:
		print('[*] Get the wanted results')
		plt.plot(
			[0, 1], [0, 1], linestyle='--', lw=2,
			color=colors[-1], label='Diag', alpha=.5)
		plt.rcParams['backend'] = 'TkAgg'
		plt.rcParams['font.sans-serif'] = ['SimHei']
		plt.ylim(0, 1)
		plt.xlim(0, 1)
		plt.xlabel("FPR")
		plt.ylabel('TPR')
		plt.legend(loc="lower right")
		plt.title(f'ROC curve - {model_type.upper()}')
		plt.show()
	# plt.savefig('ROC_curve.png')
	else:
		print('-' * 5)

	plt.clf()

	# Draw Confidence
	acc_list, threshold_list = draw_confidence_2(whole_truth, whole_probability)

	return (flag, whole_misclassified_indices,
	        (mean_auc, f1_all, revised_f1_all, acc_all, revised_acc_all),
	        acc_list, threshold_list)



if __name__ == '__main__':
	data_dir = r'../../data'
	data_name = r'data_gt_256_suv_ns_t.xlsx'
	data_path = os.path.join(data_dir, data_name)
	data_linc_path = r'../../data/clinic.xlsx'
	label_name = 'trg23'
	n_fold = 5
	pca_components = 7
	auc_threshold = 0.90
	n_repeat = 100
	t_test_method = [t_test_feature, t_test_feature2][1]
	select_method = 'pca'
	model_type = ['lr', 'xgb', 'svm', 'rf'][1]

	data_label1, data_label0, raw_data = get_data(data_path, data_linc_path, label_name)
	input_data = raw_data[raw_data.columns[2:]]

	mean_list_dict = {}
	std_list_dict = {}
	start_time = time.time()
	for t in range(2):
		t_test_method = [t_test_feature, t_test_feature2][t]
		for n in range(2, 10):
			pca_components = n
			for model_type in ['lr', 'xgb', 'svm', 'rf']:
				# Feature Selection
				if select_method == 'lasso':
					t_feature = t_test_method(data_label1, data_label0, raw_data)
					while 1:
						x_selected, y_selected, _, _ = lasso_test_feature(
							data_label1, data_label0, t_feature)
						feature_names = list(x_selected.columns.values)
						if len(feature_names) > 0: break
					x_selected = raw_data.filter(items=feature_names)
					y_selected = raw_data['lab']
				elif select_method == 'pca':
					t_feature = t_test_method(data_label1, data_label0, raw_data)
					input_data = raw_data[t_feature]
					pca = PCA(n_components=pca_components)
					pca.fit(input_data)
					explained_variance_ratio = pca.explained_variance_ratio_
					selected_components = pca.components_
					x_selected = pd.DataFrame(pca.fit_transform(input_data))
					y_selected = raw_data['lab']
				else:
					assert TypeError('Wrong select method !')

				# Test Features in Different Models
				metric_lists = []
				acc_lists = []
				threshold_lists = None
				for i in range(n_repeat):
					_, _, metric_list, acc_list, threshold_list = traditional_method_cv(
						x_selected, y_selected, n_fold, auc_threshold, model_type)

					if threshold_lists is None: threshold_lists = threshold_list
					acc_lists.append(acc_list)
					metric_lists.append(metric_list)
					print(f'[{i + 1} / {n_repeat}]')

				metric_lists = np.array(metric_lists)
				mean_list, std_list = np.mean(metric_lists, axis=0), np.std(metric_lists, axis=0)
				print(f'[{select_method} --- {model_type} --- Repeat number: {n_repeat}]')
				print('Mean AUC\t\t\tF1\t\t\tRevised F1\t\t\tAcc\t\t\tRevised Acc')
				print('\t\t\t'.join([f'{m: .4f}' for m in mean_list]))
				print('\t\t\t'.join([f'{s: .4f}' for s in std_list]))
				print('\t\t\t'.join([f'{m: .4f} ± {s: .4f}' for m, s in zip(mean_list, std_list)]))

				mean_list_dict[f'{t}-{n}-{model_type}'] = mean_list
				std_list_dict[f'{t}-{n}-{model_type}'] = std_list

	for key in mean_list_dict:
		print(key + '\t\t\t'.join([f'{m: .4f} ±{s: .4f}' for m, s in zip(
			mean_list_dict[key], std_list_dict[key])]))

	workbook = openpyxl.Workbook()
	sheet = workbook.active
	for key in mean_list_dict:
		sheet.append([key] + [f'{m: .4f} ±{s: .4f}' for m, s in zip(mean_list_dict[key], std_list_dict[key])])

	workbook.save(os.path.join(
		r'E:\xai-omics\27-FC\data\results',
		data_name.split('.xlsx')[0] + '_pca.xlsx'))


	end_time = time.time()
	print(end_time - start_time, "s")
# acc_lists = np.array(acc_lists)
# acc_lists = acc_lists.T
# draw_boxplot(acc_lists.tolist(), list(map(str, np.round(threshold_lists, 2))), 'Confidence')
