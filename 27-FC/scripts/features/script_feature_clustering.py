import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelBinarizer



def fix_data(data_path, data_linc_path, label_name='trg23'):
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
	n_count = 40
	acc_dict = {}

	df_img_pos1, df_img_neg0, df_merge1 = fix_data(
		data_path, data_linc_path, label_name)

	lr_feature, xgb_feature = get_features()

	for i, feature_names in enumerate(lr_feature + xgb_feature):
		if feature_names in lr_feature: model_type = 'lr'
		else: model_type = 'xgb'

		x_lasso1 = df_merge1.filter(items=feature_names)
		y_lasso1 = df_merge1['lab']
		y_lasso1 = y_lasso1.to_numpy()
		acc_list = []

		for j in tqdm(range(n_count)):
			kmeans = KMeans(n_clusters=2)
			kmeans.fit(x_lasso1)
			cluster_labels = kmeans.labels_

			acc = np.sum(y_lasso1 == cluster_labels) / len(y_lasso1)
			if acc < 0.5: acc = 1 - acc
			acc_list.append(acc)

		acc_dict[f'{model_type}_{i + 1}'] = [
			acc_list, [np.mean(acc_list), np.std(acc_list), np.max(acc_list), np.min(acc_list)]]

	for key in acc_dict:
		print('*' * 100)
		print(f'[{key}]')
		print(acc_dict[key][0])
		print(acc_dict[key][1])