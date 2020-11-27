#' Get mean coordinates of all patches from a folder #'
#' (Description paragraph)
#' @param season The season of this scene.
#' @param scene_id The scene index.
#' @param dat The data type of options ('lc','s1','s2')
#' @return A vector of (longitude, latitude).
#' @details (Details section)
#' @export
#' @examples
#' # (Examples section)
#'  #'
coords_folder = function(season,
                        scene_id,
                        dat = 'lc'){
  folder = paste0(season_dir(season),'/',dat,'_',scene_id)
  files = list.files(folder) 
  
  files = sample(files, 5)
  y = sapply(files, function(x){
    x = paste0(folder,'/',x) # the directory of one .tif file
    return(coords_patch(x))
  }) %>% unlist()
  
  y = matrix(y,ncol = 2,byrow = TRUE) # reshape the output so that two colunums represent
  # longtitude and latitude, respectively
  cat(season,'-',scene_id,' done.\n', sep = '')
  return(apply(y, 2, mean))
}