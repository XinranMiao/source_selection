#' Get season directories#'
#' (Description paragraph)
#' @param season The season.
#' @param base_dir The base directory.
#' @return A directory of files in this season.
#' @details (Details section)
#' @export
#' @examples
#' # (Examples section)
#'  #'
season_dir = function(season,base_dir = '/Volumes/ksankaran/Sen12MS'){
  if(season=='spring'){
    return(paste0(base_dir,'/','ROIs1158_spring'))
  }else if(season == 'summer'){
    return(paste0(base_dir,'/','ROIs1868_summer'))
  }else if(season == 'fall'){
    return(paste0(base_dir,'/','ROIs1970_fall'))
  }else if(season == 'winter'){
    return(paste0(base_dir,'/','ROIs2017_winter'))
  }else{return(NA)}
}