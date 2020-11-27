#' Get coordinates from a geographical patch #'
#' (Description paragraph)
#' @param filename The directory of the geographical patch.
#' @return A vector of (longitude, latitude).
#' @details (Details section)
#' @export
#' @examples
#' # (Examples section)
#'  #'

coords_patch = function(filename){
  x <- stack(filename)
  wgs84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
  x_ <- projectExtent(x, crs=wgs84)
  
  # get the center location, to plug into google maps
  e <- extent(x_)
  long = (e@xmin + e@xmax) / 2
  lat = (e@ymin + e@ymax) / 2
  return(c(long,lat))
}