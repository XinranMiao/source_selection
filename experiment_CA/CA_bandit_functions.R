library(dplyr)
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

pred_func2 = function(source,  val, mod = 'loglm'){
  if(mod == 'lm'){
    model = lm(Value ~ ., source)
    pred = predict(model,val)
    
    mse = mean((val$Value - pred)^2) 
    return(mse)
  }else if(mod == 'loglm'){
    source$Value = log(source$Value)
    val$Value = log(val$Value)
    model = lm(Value ~ ., source)
    pred = predict(model,val)
    mse_in_dist = mean(model$residuals ^ 2)
    mse = mean((val$Value - pred)^2) 
    return(
      list(mse = mse,
           mse_in_dist = mse_in_dist,
           prediction = pred,
           model = model)
    )
  }# end if
}# end function



# choose target from longitude and latitude ####
#lw_long = -118.5;up_long = -117.5; lw_lat = 33.5;up_lat = 34.5
#lw_long = -122;up_long = -121; lw_lat = 36.5;up_lat = 37.5
choose_target = function(seed,housing,
                         lw_long = NA, up_long = NA, lw_lat = NA, up_lat = NA){
  if(1-is.na(c(lw_long,up_long,lw_lat,up_lat)) %>% sum < 4){
    i = 0
    idx_target = 1
    while(length(idx_target)<=20){
      lw_long = 0; up_long = 1000; lw_lat = 0; up_lat = 1000
      set.seed(seed + i)
      lw_long = runif(1,min(housing$Longitude),max(housing$Longitude))
      set.seed(seed + i); up_long = min(c(lw_long + runif(1, 0.5,2),max(housing$Longitude)))
      
      set.seed(seed + i); lw_lat = runif(1,min(housing$Latitude[(housing$Longitude > lw_long) &
                                                              (housing$Longitude < up_long)]),
                                         max(housing$Latitude[(housing$Longitude > lw_long) &
                                                                (housing$Longitude < up_long)]))
      set.seed(seed + i); up_lat = min(c(lw_lat + runif(1, 1,3),max(housing$Latitude[(housing$Longitude > lw_long) &
                                                                                       (housing$Longitude < up_long)])))
      idx_target = which((housing$Longitude > lw_long)&
                           (housing$Longitude < up_long) &
                           (housing$Latitude > lw_lat) &
                           (housing$Latitude < up_lat))
      i = i+1
    }
  }else{
    idx_target = which((housing$Longitude > lw_long)&
                         (housing$Longitude < up_long) &
                         (housing$Latitude > lw_lat) &
                         (housing$Latitude < up_lat))
  }
  
  return(idx_target)
  
}



sample_selection = function(Source, Validation, Test, 
                            seed,
                            K,
                            split_criteria = 'cluster', # 'cluster' or 'random'
                            output_dir,
                            num_initial = 100,
                            step = 100,
                            num_rounds = 200,
                            remove_bad = TRUE){
  set.seed(seed)
  idx_used = sample(1:nrow(Source),num_initial,replace = TRUE)
  source_in_use = Source[idx_used,]
  models = list()
  p = pred_func2(rbind(source_in_use,Validation) %>% select(-Cluster),Test,mod = 'loglm')
  err_initial = p[[1]]
  models[[1]] = p$model
  predictions = matrix(nrow = nrow(Test),ncol = num_rounds)
  
  
  idx = list()
  alpha = rep(1,K);beta = rep(1,K)
  Clusters = rep(0,num_rounds); errs = rep(0,num_rounds); errs_in_dist = rep(0, num_rounds)
  
  for (t in 1:num_rounds) {
    set.seed(t+seed * K+1)
    pi = c()
    for (j in 1:K) {
      pi[j] = rbeta(1,alpha[j],beta[j])
    }
    set.seed(t+seed * K+1)
    
    
    if(split_criteria == 'cluster'){
      Clusters[t] = which.max(pi)
      #print(pi)
      #print(alpha)
    }else if(split_criteria == 'random'){
      Clusters[t] = sample(1:K,1)
    }
    
    choice = which(Source$Cluster == Clusters[t])
    set.seed(t-seed - 2* K+1)
    idx[[t]] = sample( choice,step,replace = TRUE)
    idx_used = c(idx_used,unlist(idx[[t]]))
    
    source_in_use = rbind(source_in_use, Source[idx[[t]],])
    
    p = pred_func2(rbind(source_in_use,Validation) %>%
                     select(-Cluster),Test,mod = 'loglm')
    
    
    errs[t] = unlist(p[1])
    predictions[,t] = p$prediction
    errs_in_dist[t] = unlist(p$mse_in_dist)
    models[[t+1]] = p$model
    #dif = ifelse(t == 1, errs_in_dist[t] - err_initial,errs_in_dist[t] - errs_in_dist[t-1])
    dif = ifelse(t == 1, errs[t] - err_initial,errs[t] - errs[t-1])
    #cat('difference is ',dif,'\n')
    if(dif< -0.0001){alpha[Clusters[t]] = alpha[Clusters[t]] + 1
    }else if(dif > 0.0001){
      beta[Clusters[t]] = beta[Clusters[t]] + 1
      if(remove_bad){
        source_in_use = source_in_use[1:(nrow(source_in_use) - step-1),]
        idx_used = idx_used[1:(step * (t-1))]
      }
    }else{
      if(remove_bad){
        source_in_use = source_in_use[1:(nrow(source_in_use) - step-1),]
        idx_used = idx_used[1:(step * (t-1))]
      }
    }
  }
  
  
  results= data.frame(cluster = Clusters,
                                error = errs)
  write.csv(results,paste0(output_dir,'results_',split_criteria,'.csv'),row.names = FALSE)
  jpeg(paste0(output_dir,'bandit_error_',split_criteria,'.jpeg'));
  p_err(errs, Clusters, title =paste0('Ultimate summary statistic is ',
                                      round(summary_stat(table(Clusters) / length(Clusters) ) ,3)),
                                                              K)
  dev.off()
  return(list(results = results,
              predictions = predictions))
  
  
}


random_vs_bandit_vis = function(results_random,
                                results_selected,
                                output_dir){
  rbind(results_random %>% mutate(Method = rep('Random',nrow(results_random))),
        results_selected%>% mutate(Method = rep('Bandit',nrow(results_selected)))) %>% 
    mutate(Rounds = rep(1:nrow(results_random),2),
           Cluster = as.factor(cluster),
           Error = error) %>% 
    ggplot()+
    geom_line(aes(x =Rounds,y = Error,color= Method),size = 2)+
    #geom_point(aes(x =Rounds,y = error,color = cluster,shape = Method),size = 2) +
    theme_bw()+
    theme(legend.position = "bottom",
          axis.title = element_text(size=14,face = 'bold'),
          axis.text = element_text(size=14,face = 'bold'),
          legend.text =  element_text(size=14,face = 'bold'),
          legend.title = element_text(size=14,face = 'bold'),
          title = element_text(size = 14,face = 'bold'))
  ggsave(paste0(output_dir,'random_vs_bandit.png'),height = 10,width = 10)
  
  
  
  data.frame(Occurence  = c(unlist(table(results_random$cluster)),
                            unlist(table(results_selected$cluster))
  ),
  Method = rep(c('Random','Bandit'),each = K),
  Cluster = as.factor(rep(1:K,2)) ) %>% 
    ggplot(aes(x = Cluster, y =Occurence,fill = Method)) +
    geom_bar(stat="identity", position=position_dodge())+
    #scale_fill_brewer(palette="Paired")+
    theme_bw()+
    theme(#legend.position = "bottom",
      axis.title = element_text(size=14,face = 'bold'),
      axis.text = element_text(size=14,face = 'bold'),
      legend.text =  element_text(size=14,face = 'bold'),
      legend.title = element_text(size=14,face = 'bold'),
      title = element_text(size = 14,face = 'bold'))
  
  ggsave(paste0(output_dir,'paired_bar.png'),height = 10,width = 10)
}
