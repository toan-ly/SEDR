# Import necessary libraries
library(Seurat)
library(SeuratDisk)
library(ggplot2)
library(patchwork)
library(dplyr)
library(clevr)

options(bitmapType = 'cairo')

# Define clustering parameters
n_domains <- 10

# Define input and output directories
data_path <- "./data/Stero-seq/raw.h5ad"
save_path <- "./results/Seurat/Stereo-seq"
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

# Load raw.h5ad file
cat("Loading raw.h5ad file...\n")
raw_h5ad <- file.path(data_path)

# Convert .h5ad to Seurat-compatible .h5Seurat format
h5seurat_file <- sub(".h5ad$", ".h5Seurat", raw_h5ad)
if (!file.exists(h5seurat_file)) {
  Convert(raw_h5ad, dest = "h5Seurat", overwrite = TRUE)
}

# Load the converted file into Seurat
sp_data <- LoadH5Seurat(h5seurat_file)

# Data preprocessing: Visualization and QC plots
plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
qc_plot <- wrap_plots(plot1, plot2)
ggsave(file.path(save_path, "QC.png"), plot = qc_plot, width = 10, height = 5)

# Data normalization using SCTransform
cat("Running SCTransform...\n")
sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)

# Dimensionality reduction and clustering
cat("Performing dimensionality reduction and clustering...\n")
sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

# Find optimal resolution for clustering
sp_data <- tryCatch({
  for (resolution in seq(0.5, 0.3, by = -0.01)) {
    sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution)
    if (length(unique(sp_data@meta.data$seurat_clusters)) == n_domains) {
      cat("Optimal resolution found: ", resolution, "\n")
      break
    }
  }
  sp_data
}, error = function(e) {
  stop("Error during clustering: ", e$message)
})

# Run UMAP for visualization
sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

# Visualization of clustering results
p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE)
p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3) + ggtitle("Clustering")
clusters_plot <- p1 + p2
ggsave(file.path(save_path, "spatial_clustering.png"), plot = clusters_plot, width = 10, height = 5)

# Save results
cat("Saving results...\n")
saveRDS(sp_data, file.path(save_path, "Seurat_final.rds"))
write.table(GetAssayData(sp_data, assay = "Spatial", slot = "counts"),
            file = file.path(save_path, "expression_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(sp_data@meta.data,
            file = file.path(save_path, "cell_metadata.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

umap_coords <- as.data.frame(sp_data@reductions$umap@cell.embeddings)
write.table(umap_coords, file = file.path(save_path, "spatial_umap_coords.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("Clustering on raw.h5ad data completed successfully.\n")
