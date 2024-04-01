import pandas as pd
import numpy as np
import statsmodels.api as sm

from scipy.stats import mannwhitneyu, ttest_ind



if __name__ == "__main__":
  '''
  Single factor analysis and multiple factor analysis
  Input form of data: xlsx file or csv file
  data ->
    lab feature1  feature2  feature3  feature4  ......
    0   xxx       xxx       xxx       xxx       ......
    1   xxx       xxx       xxx       xxx       ......
    0   xxx       xxx       xxx       xxx       ......
    0   xxx       xxx       xxx       xxx       ......
    1   xxx       xxx       xxx       xxx       ......
    1   xxx       xxx       xxx       xxx       ......
    0   xxx       xxx       xxx       xxx       ......
    ... ...       ...       ...       ...       ......
   
  '''
  # Input data
  data = pd.read_csv("f_new305.csv")

  # Parameter Setting
  threshold_mul_p = 0.1
  result_dict = {}
  multivariable_features = []

  # Start
  label = data['lab']
  label0_index = np.where(label == 0)[0]
  label1_index = np.where(label == 1)[0]

  df_features = data[data.columns[1:]]
  feature_names = df_features.columns

  # Single factor analysis
  for f in feature_names:
    feature_data = df_features[f]
    feature_label0 = feature_data[label0_index]
    feature_label1 = feature_data[label1_index]

    if 'tPSA' == f:
      statistic, p_value = mannwhitneyu(feature_label0, feature_label1)
      result_dict[f] = ['median', p_value]
    else:
      statistic, p_value = ttest_ind(
        feature_label0, feature_label1, equal_var=False)
      result_dict[f] = ['mean', p_value]

    if p_value <= threshold_mul_p: multivariable_features.append(f)

  # Multiple factor analysis
  data['intercept'] = 1
  model = sm.GLM(
    data['lab'], data[multivariable_features + ['intercept']],
    family=sm.families.Gaussian()).fit()

  for d in model.summary().tables[1].data:
    feature, p_value = d[0], d[4]
    if feature not in result_dict.keys(): continue
    else:
      result_dict[feature] = result_dict[feature] + [float(p_value)]

  for f, d in result_dict.items():
    if len(d) == 2:
      method = d[0]
      p_uni = '<0.001' if d[1] < 0.001 else f'{d[1]: .3f}'
      print(f'{f: <15}{method: <15}{p_uni: <15}')
    else:
      method = d[0]
      p_uni = '<0.001' if d[1] < 0.001 else f'{d[1]: .3f}'
      p_mul = '<0.001' if d[2] < 0.001 else f'{d[2]: .3f}'
      print(f'{f: <15}{method: <15}{p_uni: <15}{p_mul: <15}')
