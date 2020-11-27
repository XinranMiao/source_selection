#' Subsampling the Sen12MS dataset #'
#' (Description paragraph)
#' @param n The directory of the geographical patch.
#' @return A dataframe of metadata.
#' @details (Details section)
#' @export
#' @examples
#' # (Examples section)
#'  #'

subsample = function(n,n_per_scene,dirs){
  scenes <- list()
  for (i in seq_along(dirs)) { # a typo
    scenes[[i]] <- sample(list.files(dirs[i], "*tif", full.names = TRUE), n_per_scene)
  }
  metadata <- lapply(unlist(scenes), parse_path)
  metadata <- do.call(rbind, metadata)
  metadata <- metadata %>%
    mutate(
      label_path = str_replace_all(image_path, "s2", "lc"),
      label_basename = str_replace_all(image_basename, "s2", "lc")
    ) %>%
    select(image_basename, label_basename, roi, season, scene, patch, image_path, label_path)
  
  metadata = metadata %>% mutate(label_basename = sapply(metadata$label_basename, gsub, pattern = 'Ilc0',replacement = 'Is20' ),
                                 label_path = sapply(metadata$label_path, gsub, pattern = 'Ilc0',replacement = 'Is20' ))
  metadata$longitude = c()
  metadata$latitude = c()
  metadata[,c('longditude','latitude')] = sapply(metadata$label_path,
                                                 GeoR::coords_patch)%>% unlist %>% matrix(ncol = 2,byrow=TRUE)
  
  
  return(metadata)
  
}
