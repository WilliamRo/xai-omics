import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import umap

from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path
from bcp.bcp_agent import BCPAgent
from bcp.bcp_set import BCPSet
from xomics import MedicalImage
from xomics.gui.dr_gordon import DrGordon
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
from collections import OrderedDict



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
# random.seed(0)
t_file_path = r'E:\xai-omics\23-BCP\01_unet\checkpoints\0922_unet(1-4-2-1-lrelu-mp)\0922_unet(1-4-2-1-lrelu-mp).py'
t_file_path = r'E:\xai-omics\23-BCP\02_cnn\checkpoints\0925_cnn(1-5-5-lrelu)\0925_cnn(1-5-5-lrelu).py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = BCPAgent.load()

features, labels = [], []
for ds in (train_set, val_set, test_set):
  features.append(ds.features)
  labels = labels + ds.data_dict['labels']
features = np.concatenate(np.array(features), axis=0)
dataset = BCPSet(features=features, targets=features, name='dataset')
# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from bcp_core import th
th.val_batch_size = 4
th.eval_batch_size = 4
th.random_noise = 1
th.random_rotation = 1
th.random_flip = 1
model: Predictor = th.model()

z_tensor = model.layers[4].output_tensor
reconstruct_tensor = model.layers[-1].output_tensor

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
tsne_accuracy, umap_accuracy = [], []
tsne_misclass, umap_misclass = OrderedDict(), OrderedDict()
round_len = 25
point_size = 80
threshold = 0.8
num_clusters = 2
repetition = 30
for i in range(round_len):
  reconstruct_images, z_features = [], []
  for r in range(repetition):
    print('-'*5, r, '-'*5)
    # reconstruct_images.append(model.evaluate(reconstruct_tensor, dataset, batch_size=1, verbose=True))
    z_features.append(model.evaluate(z_tensor, dataset, batch_size=1, verbose=True))

  # reconstruct_images = np.sum(reconstruct_images, axis=0) / n
  z_features = np.sum(z_features, axis=0) / repetition

  # -----------------------------------------------------------------------------
  # 5. Visualize tensor in Pictor
  # -----------------------------------------------------------------------------
  # mi_list = []
  # features = np.squeeze(dataset.features)
  # outputs = np.squeeze(reconstruct_images)
  # for f, o, id in zip(features, outputs, labels):
  #   mi: MedicalImage = MedicalImage(images={'pet': f, 'prediction': o},
  #                                   key=id)
  #   mi_list.append(mi)
  #
  # dg = DrGordon(mi_list)
  # dg.slice_view.set('vmax', auto_refresh=False)
  # dg.slice_view.set('vmin', auto_refresh=False)
  # dg.show()
  # -----------------------------------------------------------------------------
  # 6. Clustering
  # -----------------------------------------------------------------------------

  real_labels = np.array([l.split('-')[-1] for l in labels])
  pids = [l.split('-')[0] for l in labels]
  e_t_n = {'left': 0, 'right': 0, 'both': 0, 'normal': 3}
  real_labels = np.vectorize(e_t_n.get)(real_labels)
  print(np.sum(real_labels == 0))
  print(np.sum(real_labels == 1))
  print(np.sum(real_labels == 2))
  print(np.sum(real_labels == 3))

  data_matrix = z_features.reshape(z_features.shape[0], -1)

  # # Elbow Method
  # inertia = []
  # k_num = 20
  # for k in range(1, k_num):
  #   kmeans = KMeans(n_clusters=k, random_state=0)
  #   kmeans.fit(data_matrix)
  #   inertia.append(kmeans.inertia_)
  #
  # plt.figure(figsize=(8, 4))
  # plt.plot(range(1, k_num), inertia, marker='o')
  # plt.title('Elbow Method')
  # plt.xlabel('Number of Clusters')
  # plt.ylabel('Inertia')
  # plt.show()

  # t-SNE or UMAP clustering
  tsne_model = TSNE(n_components=2, perplexity=5)
  umap_model = umap.UMAP(n_components=2, n_neighbors=5)
  kmeans = KMeans(n_clusters=num_clusters)
  cluster_models = [tsne_model, umap_model]

  for cm in cluster_models:
    if isinstance(cm, TSNE):
      tool = 't-SNE'
    elif isinstance(cm, umap.UMAP):
      tool = 'UMAP'
    else:
      assert TypeError

    pred_data = cm.fit_transform(data_matrix)
    pred_labels = kmeans.fit_predict(pred_data)

    real_labels[real_labels != 0] = 1
    pred_result = np.logical_not(np.logical_xor(pred_labels, real_labels))
    if np.sum(pred_result == True) < len(pred_result) * 0.5:
      pred_result = np.logical_not(pred_result)
      pred_labels = np.logical_not(pred_labels)

    # Calculate the Accuracy
    acc = np.sum(pred_result == True) / len(pred_result)

    # Plot
    if acc >= threshold:
      fig, axe = plt.subplots(1, 2, figsize=(20, 10))

      scatter0 = axe[0].scatter(pred_data[:, 0], pred_data[:, 1],
                                c=pred_labels, cmap='viridis', s=point_size)
      axe[0].set_title(f'{tool} - prediction - acc:{round(acc, 3)}')

      scatter1 = axe[1].scatter(pred_data[:, 0], pred_data[:, 1],
                                c=real_labels, cmap='viridis', s=point_size)
      axe[1].set_title(f'{tool} - ground truth')

      for i, p in enumerate(pids):
        l = p if not pred_result[i] else ''
        axe[0].annotate(l, (pred_data[i, 0], pred_data[i, 1]),
                        textcoords="offset points", xytext=(0, 1), ha='center')
        axe[1].annotate(l, (pred_data[i, 0], pred_data[i, 1]),
                        textcoords="offset points", xytext=(0, 1), ha='center')
      plt.show()

    # Print clustering Accuracy
    print(f'{tool} --- accuracy: {round(acc, 3)}')

    indices = np.where(pred_result == False)[0]
    if tool == 't-SNE':
      tsne_accuracy.append(acc)
      for indice in indices:
        if pids[indice] in tsne_misclass.keys():
          tsne_misclass[pids[indice]] = tsne_misclass[pids[indice]] + 1
        else:
          tsne_misclass[pids[indice]] = 1
    else:
      umap_accuracy.append(acc)
      for indice in indices:
        if pids[indice] in umap_misclass.keys():
          umap_misclass[pids[indice]] = umap_misclass[pids[indice]] + 1
        else:
          umap_misclass[pids[indice]] = 1

tsne_accuracy = np.sum(tsne_accuracy) / round_len
umap_accuracy = np.sum(umap_accuracy) / round_len

tsne_misclass = dict(sorted(
  tsne_misclass.items(), key=lambda item: item[1], reverse=True))

umap_misclass = dict(sorted(
  umap_misclass.items(), key=lambda item: item[1], reverse=True))

print(f'Total Round: {round_len}')
for key, value in zip(tsne_misclass.keys(), tsne_misclass.values()):
  print(f't-SNE --- {key}: {value}')

for key, value in zip(umap_misclass.keys(), umap_misclass.values()):
  print(f'UMAP --- {key}: {value}')

print(f't-SNE --- accuracy in average: {round(tsne_accuracy, 3)}')
print(f'UMAP --- accuracy in average: {round(umap_accuracy, 3)}')

print()
