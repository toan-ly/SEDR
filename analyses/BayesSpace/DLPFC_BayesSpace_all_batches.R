library(BayesSpace)
library(ggplot2)


batch_cluster_map <- list(
  '151669' = 5, '151670' = 5, '151671' = 5, '151672' = 5,
  '151673' = 7, '151674' = 7, '151675' = 7, '151676' = 7,
  '151507' = 7, '151508' = 7, '151509' = 7, '151510' = 7
)

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
#   dlpfc <- readVisium(dir.input) 
#   dlpfc <- logNormCounts(dlpfc)

  set.seed(88)
  dec <- scran::modelGeneVar(dlpfc)
  top <- scran::getTopHVGs(dec, n = 2000)

  set.seed(66)
  dlpfc <- scater::runPCA(dlpfc, subset_row=top)

  ## Add BayesSpace metadata
  dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

  q <- n_clusters  # Number of clusters
  d <- 15  # Number of PCs

  ## Run BayesSpace clustering
  set.seed(104)
  dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium', 
                        nrep=50000, gamma=3, save.chain=TRUE)

  labels <- dlpfc$spatial.cluster

  ## View results
  clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
    scale_fill_viridis_d(option = "A", labels = 1:7) +
    labs(title="BayesSpace")

  ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)
 
  ##### save data
  write.table(colData(dlpfc), file=file.path(dir.output, 'bayesSpace.csv'), sep='\t', quote=FALSE)

}