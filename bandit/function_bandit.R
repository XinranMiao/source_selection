p_err = function(err, Clusters, title = 'Error over time',num_cluster){
  plot(1:length(err),err,type = 'l',xlab = 't',ylab = 'Error',cex.axis = 1,   cex.lab = 1.5,main = title,cex.main = 1.5,
       xlim = c(0,length(Clusters)+27))
  points(1:length(err),err, col = Clusters,pch = 16,cex = 1.5)
  legend('bottomright',legend = sapply(1:num_cluster,function(x){
    paste0('Cluster ',x,': ',table(Clusters)[x])
  }),
  col = 1: num_cluster,pch = rep(16, num_cluster))
}



# summary_stat(): compute summary statistic for a vector of weights ####
# input ---
# x: a vector of weights
summary_stat = function(x){
  x = unlist(x)
  u = rep(1 / length(x),length(x))
  return(mean(abs(u - x)))
}
