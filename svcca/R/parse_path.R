#' Parse a path to obtain information #'
#' (Description paragraph)
#' @param path The directory of the geographical patch.
#' @return image_path The path of the image
#' @return image_basename The basename of the path
#' @return roi Region of interest
#' @return season 
#' @return scene The scene index
#' @return patch The patch index
#' @details (Details section)
#' @export
#' @examples
#' # (Examples section)
#'  #'
parse_path <- function(path) {
  comps <- strsplit(basename(path), "_|\\.")[[1]]
  data.frame(
    "image_path" = path,
    "image_basename" = basename(path),
    "roi" = comps[1],
    "season" = comps[2],
    "scene" = comps[4],
    "patch" = comps[5]
  )
}
