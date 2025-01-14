import os
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import pandas as pd

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
                            homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('/home/lytq/SEDR/DeepST-main/deepst')
from DeepST import run

import warnings
warnings.filterwarnings('ignore')

def calculate_clustering_matrix(pred, gt, sample):
    cols = ['Sample', 'Score', "Metric"]
    df = pd.DataFrame(columns=cols)
    
    pca_ari = adjusted_rand_score(pred, gt)
    df = df.append(pd.Series([sample, pca_ari, "ARI"],
                             index=cols), ignore_index=True)
    
    pca_ami = adjusted_mutual_info_score(pred, gt)
    df = df.append(pd.Series([sample, pca_ami, "AMI"],
                             index=cols), ignore_index=True)
    
    pca_hcv = homogeneity_completeness_v_measure(gt, pred)
    df = df.append(pd.Series([sample, pca_hcv[0], "Homogeneity"],
                             index=cols), ignore_index=True)
    
    df = df.append(pd.Series([sample, pca_hcv[1], "Completeness"],
                             index=cols), ignore_index=True)
    
    df = df.append(pd.Series([sample, pca_hcv[2], "V_measure"],
                            index=cols), ignore_index=True)
    
    return df
    

def main():
    data_path = "/home/lytq/SEDR/data/BRCA1" #### to your path
    # sample = sys.argv[1]
    data_names = ['V1_Human_Breast_Cancer_Block_A_Section_1']


    for data_name in data_names:
        # if data_name != '151673': # Best one currently
        #     continue
        
        save_root = Path(f'/home/lytq/SEDR/results/DeepST/BRCA1/{data_name}')
        os.makedirs(save_root, exist_ok=True)

        print('Processing', data_name, '...')
        n_domains = 20
        # n_domains = sys.argv[2]

        deepen = run(
            save_path = save_root,
            task = "Identify_Domain", #### DeepST includes two tasks, one is "Identify_Domain" and the other is "Integration"
            pre_epochs = 800, ####  choose the number of training
            epochs = 1000, #### choose the number of training
            use_gpu = True)

        ###### Read in 10x Visium data, or user can read in themselves.
        adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name)

        gt_df = pd.read_csv(data_path + '/' + data_name + '/metadata.tsv', sep='\t')
        adata.layers['count'] = adata.X.toarray()

        ###### Segment the Morphological Image
        adata = deepen._get_image_crop(adata, data_name=data_name)

        ###### Data augmentation. spatial_type includes three kinds of "KDTree", "BallTree" and "LinearRegress", among which "LinearRegress"
        ###### is only applicable to 10x visium and the remaining omics selects the other two.
        ###### "use_morphological" defines whether to use morphological images.
        adata = deepen._get_augment(adata, spatial_type="LinearRegress", use_morphological=True)

        ###### Build graphs. "distType" includes "KDTree", "BallTree", "kneighbors_graph", "Radius", etc., see adj.py
        graph_dict = deepen._get_graph(adata.obsm["spatial"], distType = "BallTree")

        ###### Enhanced data preprocessing
        data = deepen._data_process(adata, pca_n_comps = 200)

        ###### Training models
        deepst_embed = deepen._fit(
            data = data,
            graph_dict = graph_dict,
        )
        # Remove Image_crop folder after training
        os.system(f'rm -r {save_root}/Image_crop')

        ###### DeepST outputs
        adata.obsm["DeepST_embed"] = deepst_embed

        ###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
        adata = deepen._get_cluster_data(adata, n_domains=n_domains, priori = True)
        print(adata)

        ###### Spatial localization map of the spatial domain
        sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=150, title='Clustering')
        plt.savefig(os.path.join(save_root, f'{data_name}_domains.png'), bbox_inches='tight', dpi=300)

        adata.obs['DeepST'] = adata.obs['DeepST_refine_domain']

        adata.write(save_root / 'result.h5ad')
        df_PC = pd.DataFrame(data=adata.obsm['DeepST_embed'], index=adata.obs.index)
        df_PC.to_csv(save_root / 'PCs.tsv', sep='\t')
        adata.obs.to_csv( save_root / 'metadata.tsv', sep='\t')

        ###### UMAP visualization
        sc.pp.neighbors(adata, use_rep='DeepST_embed', n_neighbors=10)
        sc.tl.umap(adata)

        fig, axes = plt.subplots(1,1,figsize=(4, 3))
        sc.pl.umap(adata, color='DeepST_refine_domain', ax=axes, show=False)
        axes.set_title('Clustering')

        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f'{data_name}_umap.png'), bbox_inches='tight', dpi=300)
        
        low_dim_data = pd.DataFrame(adata.obsm['image_feat'], index=adata.obs.index)
        expression_data = pd.DataFrame(adata.layers['count'], index=adata.obs.index, columns=adata.var.index)
        cell_metadata = adata.obs

        low_dim_data.to_csv(f"{save_root}/low_dim_data.csv")
        expression_data.T.to_csv(f"{save_root}/expression_matrix.csv")
        cell_metadata.to_csv(f"{save_root}/cell_metadata.csv")
        
        umap_coords = adata.obsm["X_umap"]
        spot_ids = adata.obs_names
        umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        umap_df["spot_id"] = spot_ids
        umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
        umap_df.to_csv(os.path.join(save_root, "spatial_umap_coords.csv"), index=False)
       
        print('Done', data_name)

if __name__ == '__main__':
    main()