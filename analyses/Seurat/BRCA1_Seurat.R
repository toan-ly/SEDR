# Import necessary libraries
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(mclust)
library(aricode)
library(clevr)  # For homogeneity, completeness, v-measure
options(bitmapType = 'cairo')
options(future.globals.maxSize = 1024 * 1024 * 1024) # 1 GB


# Define batch-specific clustering parameters
batch_cluster_map <- list(
  'V1_Human_Breast_Cancer_Block_A_Section_1' = 20
)

# Loop through all batches
for (sample.name in names(batch_cluster_map)) {
  cat("Processing batch:", sample.name, "\n")
  n_clusters <- batch_cluster_map[[sample.name]]

  # Define input and output directories
  dir.input <- file.path('./data/BRCA1/', sample.name)
  dir.output <- file.path('./results/Seurat/BRCA1')
  if (!dir.exists(dir.output)) {
    dir.create(dir.output, recursive = TRUE)
  }

  # Load spatial transcriptomics data
  sp_data <- tryCatch({
    Load10X_Spatial(dir.input, filename = "filtered_feature_bc_matrix.h5")
  }, error = function(e) {
    stop("Error loading spatial data for batch ", sample.name, ": ", e$message)
  })

  # Load metadata and add to the Seurat object
  df_meta <- read.table(file.path(dir.input, 'metadata.tsv'), sep = '\t')
  # sp_data <- AddMetaData(sp_data, metadata = df_meta$layer_guess, col.name = 'layer_guess')

  # Data processing: Visualization and QC plots
  plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
  plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
  qc_plot = wrap_plots(plot1, plot2)
  ggsave(file.path(dir.output, 'QC.png'), plot = qc_plot, width = 10, height = 5)

  # Data normalization using SCTransform
  sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)

  # Dimensionality reduction and clustering
  sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
  sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

  # Find optimal resolution for clustering
  sp_data <- tryCatch({
    for (resolution in seq(1.2, 0.4, by = -0.01)) {
      sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution)
      if (length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters) {
        cat("Optimal resolution found for batch", sample.name, ": ", resolution, "\n")
        break
      }
    }
    sp_data
  }, error = function(e) {
    stop("Error during clustering for batch ", sample.name, ": ", e$message)
  })

  # Run UMAP for visualization
  sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

  # Visualization of clustering results
  p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE) 
  p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3) + ggtitle("Clustering")
  clusters_plot <- p1 + p2
  ggsave(file.path(dir.output, 'spatial_clustering.png'), plot = clusters_plot, width = 10, height = 5)

  # Save Seurat object and results
  saveRDS(sp_data, file.path(dir.output, 'Seurat_final.rds'))
  write.table(sp_data@reductions$pca@cell.embeddings,
              file = file.path(dir.output, 'low_dim_data.tsv'),
              sep = '\t', quote = FALSE, col.names = NA)
  write.table(sp_data@meta.data,
              file = file.path(dir.output, 'cell_metadata.tsv'),
              sep = '\t', quote = FALSE, col.names = NA)

  expression_data <- as.data.frame(as.matrix(GetAssayData(sp_data, assay = "Spatial", slot = "counts")))
  write.table(t(expression_data), file = file.path(dir.output, "expression_matrix.tsv"), sep = "\t", quote = FALSE, col.names = NA)

  umap_coords <- as.data.frame(sp_data@reductions$umap@cell.embeddings)
  umap_coords$spot_id <- rownames(umap_coords)
  write.table(umap_coords, file = file.path(dir.output, "spatial_umap_coords.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

  # Evaluate clustering performance
  # gt <- sp_data@meta.data$layer_guess
  # pred <- sp_data@meta.data$seurat_clusters
  # clustering_results <- calculate_metrics(gt, pred)
  # cat("ARI for batch", sample.name, ":", ari, "\n")

  # metrics <- data.frame(ARI = clustering_results$ARI, 
  #                       AMI = clustering_results$AMI,
  #                       Homogeneity = clustering_results$Homogeneity,
  #                       Completeness = clustering_results$Completeness, 
  #                       V_Measure = clustering_results$V_Measure)
  # write.table(metrics, file = file.path(dir.output, 'clustering_metrics.tsv'), sep = '\t', quote = FALSE, row.names = FALSE)
}
cat("All batches processed successfully.\n")
