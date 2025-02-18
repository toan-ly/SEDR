library(BayesSpace)
library(ggplot2)
library(Seurat)
library(SingleCellExperiment)
library(mclust)
library(aricode)
library(clevr)  # For homogeneity, completeness, v-measure

batch_cluster_map <- list(
  # '151669' = 5, '151670' = 5, '151671' = 5, '151672' = 5,
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


data_path <- file.path("./data/DLPFC_new")
save_path <- file.path("./results/BayesSpace/DLPFC")

for (sample.name in names(batch_cluster_map)) {
  cat("Processing batch:", sample.name, "\n")
  n_clusters <- batch_cluster_map[[sample.name]]

  dir.input <- file.path(data_path, sample.name)
  dir.output <- file.path(save_path, sample.name)

  if(!dir.exists(file.path(dir.output))){
    dir.create(file.path(dir.output), recursive = TRUE)
  }

  dlpfc <- getRDS("2020_maynard_prefrontal-cortex", sample.name)
  ### load data
  # dlpfc <- readVisium(dir.input) 
  dlpfc_temp <- read10Xh5(dir.input)
  dlpfc_temp <- dlpfc_temp[, match(colnames(dlpfc), colnames(dlpfc_temp))]

  # Match barcodes
  match_idx <- match(dlpfc_temp$barcode, dlpfc$barcode)
  dlpfc$pxl_col_in_fullres <- dlpfc_temp$pxl_col_in_fullres[match_idx]
  dlpfc$pxl_row_in_fullres <- dlpfc_temp$pxl_row_in_fullres[match_idx]

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
  # labels <- dplyr::recode(dlpfc$spatial.cluster, 3, 4, 5, 6, 2, 7, 1)
  gt <- dlpfc$layer_guess
  metrics <- calculate_metrics(gt, labels)
  cat("ARI for batch", sample.name, ":", metrics$ARI, "\n")

  cluster_results <- data.frame(ARI = metrics$ARI,
                                AMI = metrics$AMI,
                                Homogeneity = metrics$Homogeneity,
                                Completeness = metrics$Completeness,
                                V_Measure = metrics$V_Measure)
  write.table(cluster_results, file = file.path(dir.output, 'clustering_results.tsv'), sep = '\t', quote = FALSE, row.names = FALSE)
  
  ## View results
  clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
    scale_fill_viridis_d(option = "A", labels = 1:7) +
    labs(title=paste("ARI =", round(metrics$ARI, 2))) +
    # theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 16))

  ggsave(file.path(dir.output, 'spatial_clustering.png'), width=5, height=5)
 
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
