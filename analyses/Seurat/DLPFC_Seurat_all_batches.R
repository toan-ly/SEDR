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

# Define batch-specific clustering parameters
batch_cluster_map <- list(
  '151669' = 5, '151670' = 5, '151671' = 5, '151672' = 5,
  '151673' = 7, '151674' = 7, '151675' = 7, '151676' = 7,
  '151507' = 7, '151508' = 7, '151509' = 7, '151510' = 7
)

# Function to calculate entropy metrics
calculate_metrics <- function(ground_truth, clusters) {
  tryCatch({
    # Remove NA values
    valid_indices <- !is.na(clusters) & !is.na(ground_truth)
    clusters <- clusters[valid_indices]
    ground_truth <- ground_truth[valid_indices]

    if (length(ground_truth) != length(clusters)) {
      stop("Mismatch in length between ground_truth and clusters.")
    }
    if (any(is.na(ground_truth)) || any(is.na(clusters))) {
      stop("Missing values found in ground_truth or clusters.")
    }

    # # Convert to factors
    # ground_truth <- as.factor(ground_truth)
    # clusters <- as.factor(clusters)

    ARI <- adjustedRandIndex(ground_truth, clusters)
    AMI <- AMI(ground_truth, clusters)
    homogeneity <- clevr::homogeneity(ground_truth, clusters)
    completeness <- clevr::completeness(ground_truth, clusters)
    v_measure <- clevr::v_measure(ground_truth, clusters)
    return(list(ARI = ARI, AMI = AMI, Homogeneity = homogeneity, Completeness = completeness, V_Measure = v_measure))
  }, error = function(e) {
    warning("Error calculating metrics: ", e$message)
    return(list(ARI = NA, AMI = NA, Homogeneity = NA, Completeness = NA, V_Measure = NA))
  })
}

# Loop through all batches
for (sample.name in names(batch_cluster_map)) {
  cat("Processing batch:", sample.name, "\n")
  n_clusters <- batch_cluster_map[[sample.name]]

  # Define input and output directories
  dir.input <- file.path('./data/DLPFC_new/', sample.name)
  dir.output <- file.path('./results/Seurat/DLPFC', sample.name)
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
  df_meta <- read.table(file.path(dir.input, 'metadata.tsv'))
  sp_data <- AddMetaData(sp_data, metadata = df_meta$layer_guess, col.name = 'layer_guess')

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
    for (resolution in seq(0.5, 0.3, by = -0.01)) {
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
  gt <- sp_data@meta.data$layer_guess
  pred <- sp_data@meta.data$seurat_clusters
  clustering_results <- calculate_metrics(gt, pred)
  cat("ARI for batch", sample.name, ":", ari, "\n")

  metrics <- data.frame(ARI = clustering_results$ARI, 
                        AMI = clustering_results$AMI,
                        Homogeneity = clustering_results$Homogeneity,
                        Completeness = clustering_results$Completeness, 
                        V_Measure = clustering_results$V_Measure)
  write.table(metrics, file = file.path(dir.output, 'clustering_metrics.tsv'), sep = '\t', quote = FALSE, row.names = FALSE)
}
cat("All batches processed successfully.\n")
