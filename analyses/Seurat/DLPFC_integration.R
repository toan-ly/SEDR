# Load necessary libraries
library(Seurat)
library(ggplot2)
library(dplyr)
library(patchwork)

# Define the list of sample names
sample_names <- c('151669', '151670', '151671', '151672', '151673', '151674', 
                  '151675', '151676', '151507', '151508', '151509', '151510')

# Initialize a list to store Seurat objects
seurat_list <- list()

# Load each dataset and store in the list
for (sample_name in sample_names) {
  # Define the directory containing the Seurat object
  dir_output <- file.path('./results/Seurat/DLPFC', sample_name)
  
  # Load the Seurat object
  seurat_obj <- readRDS(file.path(dir_output, 'Seurat_final.rds'))
  
  # Add sample identifier to metadata
  seurat_obj$sample <- sample_name
  
  # Append to the list
  seurat_list[[sample_name]] <- seurat_obj
}

# Normalize and identify variable features for each dataset
seurat_list <- lapply(seurat_list, function(obj) {
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
  return(obj)
})

# Select integration features
features <- SelectIntegrationFeatures(object.list = seurat_list)

# Find integration anchors
anchors <- FindIntegrationAnchors(object.list = seurat_list, anchor.features = features)

# Integrate data
integrated_data <- IntegrateData(anchorset = anchors)

# Switch to integrated assay
DefaultAssay(integrated_data) <- "integrated"

# Scale data and perform PCA
integrated_data <- ScaleData(integrated_data, verbose = FALSE)
integrated_data <- RunPCA(integrated_data, npcs = 30, verbose = FALSE)

# Run UMAP and clustering
integrated_data <- RunUMAP(integrated_data, reduction = "pca", dims = 1:30)
integrated_data <- FindNeighbors(integrated_data, reduction = "pca", dims = 1:30)
integrated_data <- FindClusters(integrated_data, resolution = 0.5)

# Visualization
p1 <- DimPlot(integrated_data, reduction = "umap", group.by = "sample", label = TRUE, repel = TRUE) + ggtitle("UMAP by Sample")
p2 <- DimPlot(integrated_data, reduction = "umap", group.by = "seurat_clusters", label = TRUE, repel = TRUE) + ggtitle("UMAP by Cluster")
combined_plot <- p1 + p2

# Save the plot
ggsave(file.path('./results/Seurat/DLPFC', 'Integrated_UMAP.png'), plot = combined_plot, width = 12, height = 6)

# Save the integrated Seurat object
saveRDS(integrated_data, file = file.path('./results/Seurat/DLPFC', 'Integrated_Seurat_Object.rds'))
