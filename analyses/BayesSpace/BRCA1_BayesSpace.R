library(BayesSpace)
library(ggplot2)
library(Seurat)
library(SingleCellExperiment)
library(mclust)
library(aricode)
library(clevr)  # For homogeneity, completeness, v-measure

batch_cluster_map <- list(
  'V1_Human_Breast_Cancer_Block_A_Section_1' = 20
)

data_path <- file.path("./data/BRCA1")
save_path <- file.path("./results/BayesSpace/BRCA1")

for (sample.name in names(batch_cluster_map)) {
  cat("Processing batch:", sample.name, "\n")
  n_clusters <- batch_cluster_map[[sample.name]]

  dir.input <- file.path(data_path, sample.name)
  dir.output <- file.path(save_path)
  if(!dir.exists(file.path(dir.output))){
    dir.create(file.path(dir.output), recursive = TRUE)
  }

#   dlpfc <- getRDS("2020_maynard_prefrontal-cortex", sample.name)
  ### load data
#   dlpfc <- readVisium(dir.input) 
  dlpfc <- read10Xh5(dir.input)
  dlpfc <- scuttle::logNormCounts(dlpfc)

#   dlpfc_temp <- dlpfc_temp[, match(colnames(dlpfc), colnames(dlpfc_temp))]

  # Match barcodes
#   match_idx <- match(dlpfc_temp$barcode, dlpfc$barcode)
#   dlpfc$pxl_col_in_fullres <- dlpfc_temp$pxl_col_in_fullres[match_idx]
#   dlpfc$pxl_row_in_fullres <- dlpfc_temp$pxl_row_in_fullres[match_idx]

  # dlpfc <- scuttle::logNormCounts(dlpfc)
  # dlpfc <- Load10X_Spatial(dir.input, filename = "filtered_feature_bc_matrix.h5")
  # sce <- as.SingleCellExperiment(dlpfc)


  set.seed(101)
  dec <- scran::modelGeneVar(dlpfc)
  top <- scran::getTopHVGs(dec, n = 2000)

  set.seed(102)
  dlpfc <- scater::runPCA(dlpfc, subset_row=top)

  ## Add BayesSpace metadata
  dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

  q <- n_clusters  # Number of clusters
  d <- 15  # Number of PCs

  ## Run BayesSpace clustering
  set.seed(104)
  dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium', 
                        nrep=10000, gamma=3, save.chain=FALSE)


  labels <- dlpfc$spatial.cluster
  
  ## View results
  clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
    scale_fill_viridis_d(option = "A", labels = 1:20) +
    labs(title="BayesSpace") +
    theme(plot.title = element_text(hjust = 0.5, size = 16))

  ggsave(file.path(dir.output, 'spatial_clustering.png'), width=5)
 
  ##### save data
  write.table(reducedDim(dlpfc, "PCA"), 
            file = file.path(dir.output, 'low_dim_data.tsv'), 
            sep = '\t', quote = FALSE, col.names = NA)
  write.table(colData(dlpfc), 
              file=file.path(dir.output, 'cell_metadata.tsv'), 
              sep='\t', quote=FALSE, col.names = NA)
  umap_coords <- as.data.frame(reducedDim(dlpfc, "UMAP_neighbors15"))
  umap_coords$spot_id <- rownames(umap_coords)

  # Save UMAP coordinates
  write.table(umap_coords, 
              file = file.path(dir.output, "spatial_umap_coords.tsv"), 
              sep = "\t", quote = FALSE, row.names = FALSE)

  umap_plot <- ggplot(umap_coords, aes(x = V1, y = V2, color = as.factor(labels))) +
      geom_point(size = 1.5, alpha = 0.8) +
      scale_color_brewer(palette = "Set1") +
      labs(title = "UMAP", x = "UMAP 1", y = "UMAP 2", color = 'Cluster') +
      # theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 16),
            panel.grid = element_blank(),
            panel.background = element_blank(),
            axis.line = element_line(color = "black")
      )
  ggsave(file.path(dir.output, 'umap.png'), plot = umap_plot, width = 5, height = 5)

  expression_data <- as.data.frame(as.matrix(assay(dlpfc, "counts")))
  write.table(t(expression_data), 
              file = file.path(dir.output, "expression_matrix.tsv"), 
              sep = "\t", quote = FALSE, col.names = NA)

}
