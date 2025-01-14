import os 
import sys
sys.path.append('/home/lytq/SEDR/DeepST-main/deepst')
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings('ignore')

data_path = "/home/lytq/SEDR/data" 
data_name = "Stero-seq"
save_path = "/home/lytq/SEDR/results/DeepST/Stereo-seq" 
os.makedirs(save_path, exist_ok=True)
n_domains = 10 

deepen = run(save_path = save_path,
	task = "Identify_Domain", 
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True)

###### Read in other spatial data, or user can read in themselves. Including original expression
###### information and spatial location information, where the location information is saved in .obsm["spatial"]
# adata = deepen._get_adata(platform="stereoSeq", data_path=data_path, data_name=data_name)
data_root = Path('./data/Stero-seq')
if not os.path.exists(data_root / 'raw.h5ad'):
    counts = pd.read_csv(data_root / 'RNA_counts.tsv.gz', sep='\t', index_col=0).T
    counts.index = [f'Spot_{i}' for i in counts.index]
    adata = sc.AnnData(counts)
    adata.X = csr_matrix(adata.X, dtype=np.float32)

    df_pos = pd.read_csv(data_root / 'position.tsv', sep='\t')
    adata.obsm['spatial'] = df_pos[['y','x']].values

    used_barcode = pd.read_csv(os.path.join(data_root / 'used_barcodes.txt'), sep='\t', header=None)
    used_barcode = used_barcode[0]
    adata = adata[used_barcode,]

    adata.write( data_root / 'raw.h5ad')
else:
    adata = sc.read_h5ad( data_root / 'raw.h5ad')

###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
###### is only applicable to 10x visium and the remaining omics selects the other two.
###### "use_morphological" defines whether to use morphological images.
adata = deepen._get_augment(adata, spatial_type="BallTree", use_morphological=False)

###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "BallTree")

###### Enhanced data preprocessing
data = deepen._data_process(adata, pca_n_comps = 200)

###### Training models
deepst_embed = deepen._fit(
		data = data,
		graph_dict = graph_dict,)
###### DeepST outputs
adata.obsm["DeepST_embed"] = deepst_embed

###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori = True)

###### Spatial localization map of the spatial domain
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
sc.pl.spatial(adata, color='DeepST_refine_domain', title='DeepST', spot_size=40, show=False, ax=ax)
ax.invert_yaxis()
plt.savefig(os.path.join(save_path, f'spatial_clusters.png'), bbox_inches='tight', dpi=300)

###### Each cluster 
fig, axes = plt.subplots(2,5,figsize=(1.7*5, 1.5*2), sharex=True, sharey=True)
axes = axes.ravel()

for i in range(n_domains):
    sub = adata[adata.obs['DeepST_refine_domain'] == i+1]
    sc.pl.spatial(sub, spot_size=30, color='DeepST_refine_domain', ax=axes[i], legend_loc=None, show=False)
    axes[i].set_title(i)

xmin = adata.obsm['spatial'][:, 0].min()
xmax = adata.obsm['spatial'][:, 0].max()
ymin = adata.obsm['spatial'][:, 1].min()
ymax = adata.obsm['spatial'][:, 1].max()

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

plt.subplots_adjust(wspace=0, hspace=0.05)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f'{data_name}_each_cluster.png'), bbox_inches='tight', dpi=300)