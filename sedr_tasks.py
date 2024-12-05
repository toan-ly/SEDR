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
from sklearn.decomposition import PCA
from skimage.io import imread
from scipy.sparse import csr_matrix
from PIL import Image

import SEDR # Import model

warnings.filterwarnings('ignore')
Image.MAX_IMAGE_PIXELS = None

random_seed = 2023
# SEDR.fix_seed(random_seed)
torch.manual_seed(random_seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
    sub_adata = adata[
        (adata.obs['array_row'] < 33) &
        (adata.obs['array_row'] > 15) &
        (adata.obs['array_col'] < 78) &
        (adata.obs['array_col'] > 48)
    ]
    return sub_adata

def preprocess_adata(input_path):
    adata = sc.read_visium(input_path)
    adata.var_names_make_unique()
    adata.layers["count"] = adata.X.toarray()

    # Basic preprocessing
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="count", n_top_genes=2000)
    adata = adata[:, adata.var["highly_variable"] == True]
    sc.pp.scale(adata)

    # Dimension reduction
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=random_seed).fit_transform(adata.X)

    return adata

def train_model(input_path, output_dir, task_type, model_params):
    adata = preprocess_adata(input_path)

    model = SEDR()
    print(f'Training the model for {task_type}...')
    model.fit(adata.obsm["X_pca"]) 
    
    adata.obsm["SEDR"] = model.get_latent_features()
    if task_type == "imputation":
        adata.obsm["de_feat"] = model.get_reconstructed_features()
    
    return adata

def visualize_clustering(adata, task_dir, save_fig=False):
    sub_adata = adata[~pd.isnull(adata.obs["layer_guess"])]
    ARI = metrics.adjusted_rand_score(sub_adata.obs["layer_guess"], sub_adata.obs["SEDR"])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sc.pl.spatial(adata, color="layer_guess", ax=axes[0], show=False)
    sc.pl.spatial(adata, color="SEDR", ax=axes[1], show=False)
    axes[0].set_title("Manual Annotation")
    axes[1].set_title(f"Clustering: (ARI={ARI:.4f})")
    plt.tight_layout()

    if save_fig:
        plt.savefig(task_dir / "spatial_clustering.png")
    plt.close()

def visualize_imputation(adata, task_dir, save_fig=False):
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
    
    list_genes = ['BCL6','FCER2','EGR1']

    for gene in list_genes:
        idx = adata.var.index.tolist().index(gene)
        adata.obs[f'{gene}(denoised)'] = adata.obsm['de_feat'][:, idx]

    newcmp = LinearSegmentedColormap.from_list('new', ['#EEEEEE','#009900'], N=1000)
    fig, axes = plt.subplots(3,2,figsize=(3*2,3*3))
    _ = 0
    for gene in ['BCL6', 'FCER2','EGR1']:
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
    meta_data = adata.obs[['batch']]

    data_mat = adata.obsm['SEDR']
    vars_use = ['batch']
    ho = hm.run_harmony(data_mat, meta_data, vars_use)

    res = pd.DataFrame(ho.Z_corr).T
    res_df = pd.DataFrame(data=res.values, columns=['X{}'.format(i+1) for i in range(res.shape[1])], index=adata.obs.index)
    adata.obsm[f'SEDR.Harmony'] = res_df
    
    sc.pp.neighbors(adata, use_rep="SEDR.Harmony", metric="cosine")
    sc.tl.umap(adata)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sc.pl.umap(adata, color=["layer_guess", "batch_name"], ax=axes, show=False)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(task_dir / "umap_batch_integration.png")
    plt.close()

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
    n_clusters = 10
    fig, ax = plt.subplots(1,1,figsize=(4*1,3))
    sc.pl.spatial(adata, color='SEDR', spot_size=40, show=False, ax=ax)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_fig:
        plt.savefig(task_dir / "stereo_seq_clusters.png")
    plt.close()
    
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
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--task",
                        type=str, 
                        required=False,
                        choices=["clustering", "imputation", "batch_integration", "stereo_seq"],
                        help="Task to perform.")
    parser.add_argument("--data", 
                        type=str, 
                        required=True,
                        help="Path to the data file.")
    args = parser.parse_args()

    if args.task is None:
        for task in TASK_DIRS.keys():
            adata = train_model(args.data, TASK_DIRS[task], task, {})
            visualize_results(adata, task, TASK_DIRS[task], save_fig=True)
    else:
        adata = train_model(args.data, TASK_DIRS[args.task], args.task, {})
        visualize_results(adata, args.task, TASK_DIRS[args.task], save_fig=True)
   
        
