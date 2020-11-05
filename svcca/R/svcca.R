
#' Simple SVCCA Implementation
#' 
#' A SVCCA implementation that assumes all neuron activations have already been
#' subsampled / rearranged into matrices. Essentially a wrapper around the CCA
#' in the PMA package. Note that this will actually be a regularized CCA, which
#' is different from the original definition.
#' 
#' @param h1 The first matrix of activations.
#' @param h2 The second matrix of activations
#' @param K The number of SVD directions to project down onto.
#' @param L The number of CCA rho[k]'s to evaluate.
#' @importFrom PMA CCA
#' @export
svcca <- function(h1, h2, K = 25, L = 8) {
  svd_1 <- svd(h1)
  svd_2 <- svd(h2)
  
  h1_red <- svd_1$u[, 1:K] %*% diag(svd_1$d[1:K])
  h2_red <- svd_2$u[, 1:K] %*% diag(svd_2$d[1:K])
  cca_res <- CCA(h1_red, h2_red, standardize = FALSE, K = L, trace = FALSE)
  list("cca" = cca_res, "rho" = mean(cca_res$cors))
}
