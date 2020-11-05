
#' Paths to Sentinel Data
#' 
#' This is a helper to extract seasonality and region information from the path
#' to a sentinel image patch in BigEarthNet.
#' 
#' @param path The full path to a BigEarth sample.
#' @return result A data.frame containing the original path and the sample's
#'   attributes.
#' @export
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