n_cluster = 4
pred_func = function(source, idx, val, mod = 'lm'){
  source_in_use = source[idx,]
  if(mod == 'lm'){
    model = lm(Value ~ ., source_in_use)
    pred = predict(model,val)
    
    mse = mean((val$Value - pred)^2) 
    return(mse)
  }# end if
}# end function


selection = function(T = 500,st = 1,step = 10, s = 0,
                     compare = 'random',Source,Validation,idx_initial){
  alpha = rep(1,n_cluster)
  beta = rep(1,n_cluster)
  set.seed(s)
  Clusters = c()
  MSE = c()
  MSE_random = c()
  idx = list()
  idx_random = list()
  
  for (t in 1:T) {
    
    pi = c()
    for (j in 1:(n_cluster )) {
      pi[j] = rbeta(1,alpha[j],beta[j])
    }
    
    if(length(which(pi == min(pi))) > 1){
      Clusters[t] = sample(which(pi == min(pi)),1)
    }else{
      Clusters[t] = which.max(pi)
    }
   
    
    if(t == 1){
      idx[[t]] = c(idx_initial,sample(which(Source$Cluster == Clusters[t]),step))
      idx_random[[t]] = c(idx_initial,sample(1:nrow(Source),step))
    }else{
      idx[[t]] = c(idx[[t-1]],sample(which(Source$Cluster == Clusters[t]),step))
      idx_random[[t]]  = c(idx_random[[t-1]],sample(1:nrow(Source),step))
    }
    MSE[t] = pred_func(Source[,-c(6,7,10,11)],idx[[t]],
                       Validation[,-c(6,7,10,11)])
    MSE_random[t] = pred_func(Source[,-c(6,7,10,11)],idx_random[[t]],
                              Validation[,-c(6,7,10,11)])
    
    # Comparison for the selection method, to determine the parameter updates for Beta
    comparison = ifelse(t==1,MSE_initial,MSE[t-1])
    
    if(MSE[t] < ifelse(compare == 'random',MSE_random[t],comparison)){
      alpha[Clusters[t]] = alpha[Clusters[t]] + st
    }else{
      beta[Clusters[t]] = beta[Clusters[t]] + st
    }
  }
  
  return(list(MSE = MSE,MSE_random = MSE_random,Clusters = Clusters,
              idx = idx, idx_random = idx_random))
}


add_counts = function(n_cluster,Clusters){
  tab = table(Clusters)
  lab = sapply(1:n_cluster, function(i){
    paste0(i,' (',tab[i],')')
  })
  return(lab)
}
p_selection = function(selection,random, Clusters, title = 'Validation MSE over time',n_cluster,
                 initial=NULL,best=NULL,ylim = c(0,1)){
  plot(1:length(selection),selection,type = 'l',xlab = 't',ylab = 'MSE',cex.axis = 1,   cex.lab = 1.5,main = title,cex.main = 1.5,
       ylim = ylim )
  points(1:length(selection),selection, col = Clusters,pch = 16,cex = 1.5)

  legend('topright',legend = add_counts(n_cluster,Clusters),col = 1: n_cluster,pch = rep(16, n_cluster))
  
  lines(random,col = 'Orange',lwd = 4)
}


