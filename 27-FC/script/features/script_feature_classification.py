from scipy import stats
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('TKAgg')  # 或者使用其他可用的后端，如 'QtAgg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb



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


def traditional_method_cv(x, y, n_splits=1, auc_threshold=0.75, model_type='lr'):
	if model_type == 'lr':
		print('\n' + '-' * 20 + 'Logistic Regression Cross Validation' + '-' * 20 + '\n')
		model = LogisticRegression()
	elif model_type == 'xgb':
		print('\n' + '-' * 20 + 'XGB Cross Validation' + '-' * 20 + '\n')
		model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
	elif model_type == 'svm':
		print('\n' + '-' * 20 + 'SVM Cross Validation' + '-' * 20 + '\n')
		model = SVC()
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

		plt.plot(fpr, tpr, label=f'Validation Set {fold_i + 1} (AUC = %0.2f)' % auc_score)

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
	# std_auc = np.std(tprs, axis=0)
	plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_auc)
	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)

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
		plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
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

	return flag, whole_misclassified_indices, (mean_auc, f1_all, revised_f1_all, acc_all, revised_acc_all)


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


def get_features():
	lr1 = ['wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1']

	lr2 = ['wavelet-LHL_firstorder_StandardDeviation',
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

	lr3 = ['wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'original_glrlm_RunLengthNonUniformity.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1']

	lr4 = ['wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'original_glrlm_RunLengthNonUniformity.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1']

	lr5 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	       'wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glcm_MCC.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	lr6 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	       'wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glcm_MCC.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	lr7 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	       'wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glcm_MCC.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	lr8 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	       'wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'original_glrlm_RunLengthNonUniformity.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glcm_MCC.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	lr9 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	       'wavelet-LHL_firstorder_StandardDeviation',
	       'wavelet-LHL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_gldm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_GrayLevelVariance',
	       'wavelet-HLL_glrlm_RunLengthNonUniformity',
	       'wavelet-HHL_glcm_Idn',
	       'original_shape_Compactness2.1',
	       'original_glrlm_GrayLevelNonUniformityNormalized.1',
	       'original_glrlm_RunLengthNonUniformity.1',
	       'log-sigma-3-mm-3D_glcm_Imc2.1',
	       'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	       'wavelet-HLL_ngtdm_Coarseness.1',
	       'wavelet-HLH_glcm_MCC.1',
	       'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	       'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	xgb1 = ['wavelet-LHL_firstorder_StandardDeviation',
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

	xgb2 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	        'wavelet-LHL_firstorder_StandardDeviation',
	        'wavelet-LHL_glrlm_GrayLevelVariance',
	        'wavelet-HLL_gldm_GrayLevelVariance',
	        'wavelet-HLL_glrlm_GrayLevelVariance',
	        'wavelet-HLL_glrlm_RunLengthNonUniformity',
	        'wavelet-HHL_glcm_Idn',
	        'original_shape_Compactness2.1',
	        'original_glrlm_GrayLevelNonUniformityNormalized.1',
	        'log-sigma-3-mm-3D_glcm_Imc2.1',
	        'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	        'wavelet-HLL_ngtdm_Coarseness.1',
	        'wavelet-HLH_glcm_MCC.1',
	        'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	        'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	        'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	xgb3 = ['log-sigma-3-mm-3D_gldm_LargeDependenceHighGrayLevelEmphasis',
	        'wavelet-LHL_firstorder_StandardDeviation',
	        'wavelet-LHL_glrlm_GrayLevelVariance',
	        'wavelet-HLL_gldm_GrayLevelVariance',
	        'wavelet-HLL_glrlm_GrayLevelVariance',
	        'wavelet-HLL_glrlm_RunLengthNonUniformity',
	        'wavelet-HHL_glcm_Idn',
	        'original_shape_Compactness2.1',
	        'original_glrlm_GrayLevelNonUniformityNormalized.1',
	        'original_glrlm_RunLengthNonUniformity.1',
	        'log-sigma-3-mm-3D_glcm_Imc2.1',
	        'wavelet-HLL_glszm_GrayLevelNonUniformityNormalized.1',
	        'wavelet-HLL_ngtdm_Coarseness.1',
	        'wavelet-HLH_glcm_MCC.1',
	        'wavelet-HLH_glrlm_ShortRunEmphasis.1',
	        'wavelet-HHL_glrlm_ShortRunEmphasis.1',
	        'wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis.1']

	return (lr1, lr2, lr3, lr4, lr5, lr6, lr7, lr8, lr9), (xgb1, xgb2, xgb3)



if __name__ == '__main__':
	data_path = r'../../data/data_with_name.xlsx'
	data_linc_path = r'../../data/clinic.xlsx'
	label_name = 'trg23'
	n_fold, n_repeat = 5, 1
	auc_threshold = 0.65
	n_count = 20
	metric_dict = {}

	df_img_pos1, df_img_neg0, df_merge = get_data(
		data_path, data_linc_path, label_name)
	lr_feature, xgb_feature = get_features()

	for i, feature_names in enumerate(lr_feature + xgb_feature):
		if feature_names in lr_feature: model_type = 'lr'
		else: model_type = 'xgb'

		metric_lists = []
		for j in range(n_count):
			print(j + 1)
			x_lasso1 = df_merge.filter(items=feature_names)
			y_lasso1 = df_merge['lab']

			_, _, metric_list = traditional_method_cv(
				x_lasso1, y_lasso1, n_fold, auc_threshold, model_type)
			metric_lists.append(metric_list)

		metric_lists = np.array(metric_lists)
		metric_dict[f'{model_type}_{i + 1}'] = [
			np.mean(metric_lists, axis=0), np.std(metric_lists, axis=0)]

	for key in metric_dict:
		print('*' * 100)
		print(f'[{key}]')
		print('Mean AUC\t\t\tF1\t\t\tRevised F1\t\t\tAcc\t\t\tRevised Acc')
		print('\t\t\t'.join(map(str, np.round(metric_dict[key][0], 4))))
		print('\t\t\t'.join(map(str, np.round(metric_dict[key][1], 4))))

	print()

