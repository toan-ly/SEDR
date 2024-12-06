import os
from pathlib import Path
import warnings
import argparse

import scanpy as sc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import torch
from tqdm import tqdm

import harmonypy as hm

from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from skimage.io import imread
from scipy.sparse import csr_matrix

from PIL import Image
import SEDR 

# Suppress warnings and set global configurations
warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

RANDOM_SEED = 2023
SEDR.fix_seed(RANDOM_SEED)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Directories for results
EXP_DIR = Path("./results")
TASK_DIRS = {
    "clustering": EXP_DIR / "Task1_Clustering",
    "imputation": EXP_DIR / "Task2_Imputation",
    "batch_integration": EXP_DIR / "Task3_BatchIntegration",
    "stereo_seq": EXP_DIR / "Task4_StereoSeq"
}
for task_dir in TASK_DIRS.values():
    task_dir.mkdir(parents=True, exist_ok=True)

def get_sub(adata):
    return adata[
        (adata.obs['array_row'] < 33) &
        (adata.obs['array_row'] > 15) &
        (adata.obs['array_col'] < 78) &
        (adata.obs['array_col'] > 48)
    ]

def preprocess_adata(adata, task_type):
    """Preprocess the AnnData object"""
    adata.layers["count"] = adata.X.toarray()
    if task_type != 'stereo_seq':
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    if task_type != 'stereo_seq':
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count", n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"] == True]
    sc.pp.scale(adata)

    # Dimension reduction
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=RANDOM_SEED).fit_transform(adata.X)
    
    if task_type != 'stereo_seq':
        graph_dict = SEDR.graph_construction(adata, 12)
    else:
        graph_dict = SEDR.graph_construction(adata, 6)

    return adata, graph_dict

def train_model(adata, graph_dict, task_type, device, use_dec=True):
    """Train the SEDR model"""
    print(f'Training the model for {task_type}...')
    if task_type == "imputation":
        model = SEDR.Sedr(adata.X, graph_dict, mode='imputation', device=device)
    else:
        model = SEDR.Sedr(adata.obsm["X_pca"], graph_dict, device=device)
    
    if use_dec:
        model.train_with_dec()
    else:
        model.train_without_dec()
    sedr_feat, _, _, _ = model.process()
    adata.obsm["SEDR"] = sedr_feat
    
    if task_type == "imputation":
        adata.obsm["de_feat"] = model.recon()
    
    print(f'Training completed for {task_type}!')
    return model, adata

def visualize_clustering(adata, task_dir, save_fig=False):
    """Visualize Task 1 - Clustering"""
    sub_adata = adata[~pd.isnull(adata.obs["layer_guess"])]
    ARI = metrics.adjusted_rand_score(sub_adata.obs["layer_guess"], sub_adata.obs["SEDR"])

    # Plot spatial clustering
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sc.pl.spatial(adata, color="layer_guess", ax=axes[0], show=False)
    sc.pl.spatial(adata, color="SEDR", ax=axes[1], show=False)
    axes[0].set_title("Manual Annotation")
    axes[1].set_title(f"Clustering: (ARI={ARI:.4f})")
    plt.tight_layout()

    if save_fig:
        plt.savefig(task_dir / "spatial_clustering.png")
    plt.close()

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='SEDR', metric='cosine')
    sc.tl.umap(adata)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    sc.pl.umap(adata, color='layer_guess', ax=axes[0], show=False)
    sc.pl.umap(adata, color='SEDR', ax=axes[1], show=False)
    axes[0].set_title('Manual Annotation')
    axes[1].set_title('Clustering')

    for ax in axes:
        ax.set_aspect(1)

    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "umap.png")
    plt.close()
    
    # Plot silhouette score
    sil_score = silhouette_score(adata.obsm['SEDR'], adata.obs['SEDR'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(adata.obs['SEDR'], bins=30, ax=ax)
    ax.set_title(f'Silhouette Score: {sil_score:.4f}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Silhouette Score')
    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "silhouette.png")
    plt.close()    

def visualize_imputation(adata, task_dir, save_fig=False):
    """Visualize Task 2 - Imputation"""
    # Plot IGHD and 3 genes with high correlation
    newcmp = LinearSegmentedColormap.from_list('new', ['#EEEEEE','#009900'], N=1000)
    genes = ["IGHD", "MS4A1", "CD1C", "CD3D"]
    for gene in genes:
        idx = adata.var.index.tolist().index(gene)
        adata.obs[f"{gene}(denoised)"] = adata.obsm["de_feat"][:, idx]

    fig, axes = plt.subplots(1, len(genes), figsize=(4 * len(genes), 4))
    axes = axes.ravel()

    for i in range(len(genes)):
        gene = genes[i]
        sc.pl.spatial(adata, color=f'{gene}(denoised)', ax=axes[i], vmax='p99', vmin='p1', alpha_img=0, cmap=newcmp, colorbar_loc=None, size=1.6, show=False)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(task_dir / "imputation_genes.png")
    plt.close()
    
    # Plot pearson correlation
    list_idx = []
    for gene in genes:
        list_idx.append(adata.var.index.tolist().index(gene))

    list_corr_raw = np.corrcoef(adata.X[:, list_idx].T)[0, 1:]
    list_corr_denoised = adata.obs[[f'{gene}(denoised)' for gene in genes]].corr().iloc[0, 1:]

    results = [
        ['raw', 'MS4A1', list_corr_raw[0]],
        ['raw', 'CD1C', list_corr_raw[1]],
        ['raw', 'CD3D', list_corr_raw[2]],
        ['SEDR', 'MS4A1', list_corr_denoised[0]],
        ['SEDR', 'CD1C', list_corr_denoised[1]],
        ['SEDR', 'CD3D', list_corr_denoised[2]],
    ]

    df_results = pd.DataFrame(data=results, columns=['method','gene','corr'])
    df_results.to_csv(task_dir / "correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(data=df_results, x='method', y='corr', hue='gene', order=['raw','SEDR'], palette='Set1')

    ax.set_xlabel('')
    ax.set_ylabel('Pearson Correlation')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "correlation.png")
    plt.close()
    
    # Plot marker genes for GCs
    list_genes = ['BCL6','FCER2','EGR1']

    for gene in list_genes:
        idx = adata.var.index.tolist().index(gene)
        adata.obs[f'{gene}(denoised)'] = adata.obsm['de_feat'][:, idx]

    newcmp = LinearSegmentedColormap.from_list('new', ['#EEEEEE','#009900'], N=1000)
    fig, axes = plt.subplots(3,2,figsize=(3*2,3*3))
    _ = 0
    for gene in list_genes:
        i = adata.var.index.tolist().index(gene)

        adata.var['mean_exp'] = adata.X.mean(axis=0)
        sorted_gene = adata.var.sort_values('mean_exp', ascending=False).index

        adata.obs['raw'] = adata.X[:, i]
        sub_adata = get_sub(adata)
        sc.pl.spatial(sub_adata, color='raw', ax=axes[_][0], vmax='p99', vmin='p1', alpha_img=0, cmap=newcmp, colorbar_loc=None, size=1.7, show=False)
        axes[_][0].set_title(gene)

        adata.obs['recon'] = adata.obsm['de_feat'][:, i]
        sub_adata = get_sub(adata)
        sc.pl.spatial(sub_adata, color='recon', ax=axes[_][1], vmax='p99', vmin='p1', alpha_img=0, cmap=newcmp, colorbar_loc=None, size=1.7, show=False)
        axes[_][1].set_title(f'De-noised {gene}')

        _ += 1

    for ax in axes.ravel():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.subplots_adjust(wspace=0.01, hspace=0.04)
    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "denoised_genes.png")
    plt.close()
    
def visualize_batch_integration(adata, task_dir, save_fig=False):
    """Visualize Task 3 - Batch Integration"""
    # Plot UMAP
    sc.pp.neighbors(adata, use_rep="SEDR.Harmony", metric="cosine")
    sc.tl.umap(adata)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sc.pl.umap(adata, color=["layer_guess", "batch_name"], show=False)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(task_dir / "umap_batch_integration.png")
    plt.close()

    # Plot LISI scores
    ILISI = hm.compute_lisi(adata.obsm['SEDR.Harmony'], adata.obs[['batch']], label_colnames=['batch'])[:, 0]
    CLISI = hm.compute_lisi(adata.obsm['SEDR.Harmony'], adata.obs[['layer_guess']], label_colnames=['layer_guess'])[:, 0]

    df_ILISI = pd.DataFrame({
        'method': 'SEDR',
        'value': ILISI,
        'type': ['ILISI']*len(ILISI)
    })

    df_CLISI = pd.DataFrame({
        'method': 'SEDR',
        'value': CLISI,
        'type': ['CLISI']*len(CLISI)
    })

    # Save LISI scores
    df_ILISI.to_csv(task_dir / "ILISI.csv", index=False)
    df_CLISI.to_csv(task_dir / "CLISI.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(4, 5))
    sns.boxplot(data=df_ILISI, x='method', y='value', ax=axes[0])
    sns.boxplot(data=df_CLISI, x='method', y='value', ax=axes[1])
    axes[0].set_ylim(1, 3)
    axes[1].set_ylim(1, 7)
    axes[0].set_title('iLISI')
    axes[1].set_title('cLISI')

    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "lisi.png")
    plt.close()
    
def visualize_stereo_seq(adata, task_dir, save_fig=False):
    fig, ax = plt.subplots(1,1,figsize=(4*1,3))
    sc.pl.spatial(adata, color='SEDR', spot_size=40, show=False, ax=ax)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "spatial_clusters.png")
    plt.close()
        
    # Each cluster
    n_clusters = 10
    fig, axes = plt.subplots(2,5,figsize=(1.7*5, 1.5*2), sharex=True, sharey=True)
    axes = axes.ravel()

    for i in range(n_clusters):
        sub = adata[adata.obs['SEDR'] == i+1]
        sc.pl.spatial(sub, spot_size=30, color='SEDR', ax=axes[i], legend_loc=None, show=False)
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
    if save_fig:
        plt.savefig(task_dir / "stereo_seq_clusters.png")
    plt.close()

def visualize_results(adata, task_type, task_dir, save_fig=False):
    if task_type == "clustering":
        visualize_clustering(adata, task_dir, save_fig)
    elif task_type == "imputation":
        visualize_imputation(adata, task_dir, save_fig)
    elif task_type == "batch_integration":
        visualize_batch_integration(adata, task_dir, save_fig)
    elif task_type == "stereo_seq":
        visualize_stereo_seq(adata, task_dir, save_fig)
        
def run_clustering():
    data_root = Path('./data/DLPFC')
    sample_name = '151673'
    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7

    # Loading data
    adata = sc.read_visium(data_root / sample_name)
    adata.var_names_make_unique()
    df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    adata.obs['layer_guess'] = df_meta['layer_guess']
    
    # Preprocessing
    adata, graph_dict = preprocess_adata(adata, 'clustering')

    # Training
    _, adata = train_model(adata, graph_dict, 'clustering', use_dec=True, device=DEVICE)
    
    # Evaluation and visualization
    SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')
    visualize_results(adata, 'clustering', TASK_DIRS['clustering'], save_fig=True)
    
def run_imputation():
    data_root = Path('./data/Human_Lymph_Node/')
    
    # Loading data
    adata = sc.read_visium(data_root)
    adata.var_names_make_unique()

    # Preprocessing
    adata, graph_dict = preprocess_adata(adata, 'imputation')
    
    # Training
    _, adata = train_model(adata, graph_dict, 'imputation', use_dec=True, device=DEVICE)
    
    # Evaluation and visualization
    visualize_results(adata, 'imputation', TASK_DIRS['imputation'], save_fig=True)
    
def run_batch_integration():
    data_root = Path('./data/DLPFC/')
    proj_list = ['151673', '151674', '151675']
    
    # Combining datasets
    for proj_name in tqdm(proj_list):
        adata_tmp = sc.read_visium(data_root / proj_name)
        adata_tmp.var_names_make_unique()

        adata_tmp.obs['batch_name'] = proj_name

        ##### Load layer_guess label, if have
        df_label = pd.read_csv(data_root / proj_name / 'metadata.tsv', sep='\t')
        adata_tmp.obs['layer_guess'] = df_label['layer_guess']
        adata_tmp= adata_tmp[~pd.isnull(adata_tmp.obs['layer_guess'])]

        graph_dict_tmp = SEDR.graph_construction(adata_tmp, 12)

        if proj_name == proj_list[0]:
            adata = adata_tmp
            graph_dict = graph_dict_tmp
            name = proj_name
            adata.obs['proj_name'] = proj_name
        else:
            var_names = adata.var_names.intersection(adata_tmp.var_names)
            adata = adata[:, var_names]
            adata_tmp = adata_tmp[:, var_names]
            adata_tmp.obs['proj_name'] = proj_name

            adata = adata.concatenate(adata_tmp)
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
            name = name + '_' + proj_name

    # Preprocessing
    adata, graph_dict = preprocess_adata(adata, 'batch_integration')

    # Training
    _, adata = train_model(adata, graph_dict, 'batch_integration', use_dec=False, device=DEVICE)
    
    # Use harmony to calculate revised PCs
    meta_data = adata.obs[['batch']]

    data_mat = adata.obsm['SEDR']
    vars_use = ['batch']
    ho = hm.run_harmony(data_mat, meta_data, vars_use)

    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
    adata.obsm[f'SEDR.Harmony'] = res_df
    
    # Evaluation and visualization
    visualize_results(adata, 'batch_integration', TASK_DIRS['batch_integration'], save_fig=True)
    
def run_stereo_seq():
    # Loading data
    data_root = Path('./data/Stero-seq/Dataset1_LiuLongQi_MouseOlfactoryBulb')
    
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
    
    # Plot before preprocessing
    adata.obs['total_exp'] = adata.X.sum(axis=1)
    fig, ax = plt.subplots()
    sc.pl.spatial(adata, color='total_exp', spot_size=40, show=False, ax=ax)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(task_dir / "total_exp.png")
    
    # Preprocessing
    adata, graph_dict = preprocess_adata(adata, 'stereo_seq')

    # Training
    _, adata = train_model(adata, graph_dict, 'stereo_seq', use_dec=True, device=DEVICE)
    
    # Evaluation and visualization
    n_clusters = 10
    SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')
    visualize_results(adata, 'stereo_seq', TASK_DIRS['stereo_seq'], save_fig=True)
    

if __name__ == "__main__":
    run_clustering()
    run_imputation()
    run_batch_integration()
    run_stereo_seq() 


        
