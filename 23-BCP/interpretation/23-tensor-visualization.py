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



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r'E:\xai-omics\23-BCP\01_unet\checkpoints\0922_unet(1-4-2-1-lrelu-mp)\0922_unet(1-4-2-1-lrelu-mp).py'
# t_file_path = r'E:\xai-omics\23-BCP\01_unet\checkpoints\0922_unet(1-3-3-lrelu)\0922_unet(1-3-3-lrelu).py'
# t_file_path = r'E:\xai-omics\23-BCP\01_unet\checkpoints\0922_unet(1-4-5-lrelu)\0922_unet(1-4-5-lrelu).py'
sys.argv.append('--developer_code=deactivate')

mod = import_from_path(t_file_path)

# th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load datas
# -----------------------------------------------------------------------------
train_set, val_set, test_set = BCPAgent.load()

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
from bcp_core import th
th.val_batch_size = 4
th.eval_batch_size = 4
model: Predictor = th.model()

z_tensor = model.layers[4].output_tensor
reconstruct_tensor = model.layers[-1].output_tensor

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
features, labels= [], []
for ds in (train_set, val_set, test_set):
  features.append(ds.features)
  labels = labels + ds.data_dict['labels']
features = np.concatenate(np.array(features), axis=0)
dataset = BCPSet(features=features, targets=features, name='dataset')

print('Evaluating!!!')
reconstruct_images = model.evaluate(reconstruct_tensor, dataset, batch_size=1, verbose=True)
z_features = model.evaluate(z_tensor, dataset, batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
mi_list = []
features = np.squeeze(dataset.features)
outputs = np.squeeze(reconstruct_images)
for f, o, id in zip(features, outputs, labels):
  mi: MedicalImage = MedicalImage(images={'pet': f, 'prediction': o},
                                  key=id)
  mi_list.append(mi)

dg = DrGordon(mi_list)
dg.slice_view.set('vmax', auto_refresh=False)
dg.slice_view.set('vmin', auto_refresh=False)
dg.show()
# -----------------------------------------------------------------------------
# 6. Clustering
# -----------------------------------------------------------------------------
real_labels = np.array([l.split('-')[-1] for l in labels])
e_t_n = {'left': 0, 'right': 1, 'both': 2, 'normal': 3}
real_labels = np.vectorize(e_t_n.get)(real_labels)

num_clusters = 4
data_matrix = z_features.reshape(z_features.shape[0], -1)

tsne = TSNE(n_components=2, perplexity=15)
tsne_data = tsne.fit_transform(data_matrix)
kmeans = KMeans(n_clusters=num_clusters)
tsne_labels = kmeans.fit_predict(tsne_data)

umap_model = umap.UMAP(n_components=2)
umap_data = umap_model.fit_transform(data_matrix)
kmeans = KMeans(n_clusters=num_clusters)
umap_labels = kmeans.fit_predict(umap_data)

fig, axe = plt.subplots(2, 2, figsize=(40, 40))

scatter0 = axe[0, 0].scatter(tsne_data[:, 0], tsne_data[:, 1], c=tsne_labels, cmap='viridis')
axe[0, 0].set_title('t-SNE')
cbar0 = plt.colorbar(scatter0, ax=axe[0, 0])
cbar0.set_label('Label')

scatter1 = axe[0, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=real_labels, cmap='viridis')
axe[0, 1].set_title('t-SNE')
cbar1 = plt.colorbar(scatter1, ax=axe[0, 1])
cbar1.set_label('Label')

scatter2 = axe[1, 0].scatter(umap_data[:, 0], umap_data[:, 1], c=umap_labels, cmap='viridis')
axe[1, 0].set_title('UMAP')
cbar2 = plt.colorbar(scatter2, ax=axe[1, 0])
cbar2.set_label('Label')

scatter3 = axe[1, 1].scatter(umap_data[:, 0], umap_data[:, 1], c=real_labels, cmap='viridis')
axe[1, 1].set_title('UMAP')
cbar3 = plt.colorbar(scatter3, ax=axe[1, 1])
cbar3.set_label('Label')

plt.show()

for id, t, u in zip(labels, tsne_labels, umap_labels):
  print(id, '---', t, '---', u)




print()
