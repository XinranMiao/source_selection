library(dplyr);library(MASS);library(bayesm); library(ggtern);library(viridis)
library(superheat)
generate_time = function(n, # simulation size
                         p =4,
                         beta_sd = rep(1,4),
                         beta_base=rep(1,4) , 
                         random = FALSE,
                         continuous = FALSE,
                         seed = 1,
                         epsilon_sd = 0.1,
                         fraction = 0.2
                         
){
  t = seq(1,10,length.out = n)
  set.seed(seed)
  X = mvrnorm(n=n,mu = rnorm(p-1,0,30),Sigma = diag(nrow = p-1))
  X = cbind(rep(1,n),X)
  set.seed(seed)
  alpha = runif(p,-1,1)
  B = matrix(nrow = 3, ncol = p)
  beta = sapply(1:length(beta_base),function(h){
    if(random == FALSE){
      if(continuous == FALSE){
        b = c(beta_base[h]*alpha[h]*t[t<3],
              beta_base[h]*(1+alpha[h])*t[(t>=3) & (t<=5)]^2,
              beta_base[h]*(alpha[h]-1)*t[t>5])
        
      }else{
        b = beta_base[h] * scale(t)
      } # continuous
    }else{ # this is random
      b = beta_base[h] * alpha[h] 
    }
    
    set.seed(seed+beta_base)
    b = b + rnorm(n = n, mean = 0, sd = beta_sd[h])
    b
  } # sapply
  ) %>% as.matrix 
  
  for (h in 1:p) {
    B[,h] = c(beta_base[h]*alpha[h], beta_base[h]*(1+alpha[h]), beta_base[h]*(alpha[h]-1)  )
  }
  
  set.seed(seed)
  y = sapply(1:n, function(i){sum(c(X[i,]) * beta[i,] ) + rnorm(1,sd=epsilon_sd)})
  dat = cbind(y,t,X)  %>% as.data.frame()
  q = quantile(t,probs = seq(0,1,by=fraction))
  names(dat) = c('y','t',sapply(0:(p-1), function(h){paste0('x',h)}) %>% unlist)
  target = dat %>% filter(t > q[order(q,decreasing = TRUE)][2])
  source = dat %>% filter(t <= q[order(q,decreasing = TRUE)][2])
  return(list(dat = dat, target = target, source = source,beta=beta,
              B=B
  )
  )
}




pred_mse = function(mod,newdata,y){
  predictions = predict(mod,newdata = newdata)
  (PMSE = mean((predictions -y)^2))
  return(PMSE)
}
w2c = function(df, # a dataframe with column 'cluster'
               w, # a vector of weights
               size = 100){
  num = c();sp_id = c()
  for (i in 1:(length(w)-1)) {
    num = c(num, round(w[i] *size)) %>% unlist
    sp_id = c(sp_id, sample(which(df$cluster == as.character(i)),
                            as.numeric(num[i]),
                            replace = TRUE))
  }
  num = c(num, size - sum(num))
  if(num[length(num)] > 0){
    sp_id = c(sp_id, sample(which(df$cluster == as.character(length(w))),
                            num[length(num)],
                            replace = TRUE))
  }
  
  return(sp_id)
}

relationship = function(t,x,alpha = 2,beta = 1/10){
  slope = (t) *beta 
  set.seed(1)
  plot(x, alpha+slope*x,ylim = c(0,20))+ rnorm(length(x),0,0.1)
  abline(a = alpha, b = slope )
}

stat = function(x){
  x = unlist(x)
  u = rep(1 / length(x),length(x))
  return(mean(abs(u - x)))
}

w2stat = function(Weights, num_cluster,num_stats = 5){
  apply(Weights[order(Weights[,num_cluster + 1])[1:num_stats],
                1:num_cluster], 
        1, stat) %>% mean
}

data2stat = function(source,target, 
                     num_cluster = 3,
                     num_simulation = 1000,
                     fraction = 0.2,
                     num_stats = 5){
  source$cluster = cut(source$t,num_cluster ,labels= 1:num_cluster)
  Weights = matrix(NA,nrow=num_simulation,ncol= num_cluster+1)
  
  for (i in 1:num_simulation) {
    Weights[i,1:num_cluster] = rdirichlet(rep(1,num_cluster))
    s = w2c(source,Weights[i,1:num_cluster],size = round(simulation_size * fraction / 15))
    model = lm(y~x, data = source[s,])
    Weights[i,num_cluster+1] = pred_mse(model,target)
  }
  
  Weights = as.data.frame(Weights); names(Weights) = c(1:num_cluster,'PMSE')
  
  stat(Weights[which.min(Weights[,num_cluster+1]),1:num_cluster])
  
  return(apply(Weights[order(Weights[,num_cluster + 1])[1:num_stats],1:num_cluster], 1, stat) %>% mean)
  
}


start2end = function(n=1000, # simulation size
                     p=4, # number of parameters
                     beta_sd ,# numeric vector of length p, sd of beta's
                     beta_base, # numeric vector of length p, 
                     seed = 1, # random seed
                     epsilon_sd = 0.1 ,# numeric value, sd of epsilon
                     K = 3 , # number of clusters
                     output_dir = './'
){
  l = generate_time(n, # simulation size
                    p ,
                    beta_sd,
                    beta_base,
                    random = 'FALSE',
                    continuous = FALSE,
                    seed ,
                    epsilon_sd =epsilon_sd )
  source = l$source; target = l$target; dat = l$dat;beta = l$beta
  write.csv(beta,paste0(output_dir,'variant_beta.csv'),row.names = FALSE)
  write.csv(dat,paste0(output_dir,'variant_dat.csv'),row.names = FALSE)
  write.csv(source,paste0(output_dir,'variant_source.csv'),row.names = FALSE)
  write.csv(target,paste0(output_dir,'variant_target.csv'),row.names = FALSE)
  source$cluster = cut(source$t,K,labels= 1:K)
  Weights = matrix(NA,nrow=num_simulation,ncol= K+1)
  
  for (i in 1:num_simulation) {
    Weights[i,1:K] = bayesm::rdirichlet(rep(1,K))
    s = w2c(df = source,
            w = Weights[i,1:K],
            size = simulation_size)
    model = lm(y~., data = source %>% dplyr::select(-cluster,-t ,-x0) %>%.[s,]) # the second column is time "t"
    Weights[i,K+1] = pred_mse(model,target %>% mutate(y = NULL,t = NULL,x0=NULL),target$y)
  }
  
  Weights = as.data.frame(Weights); names(Weights) = c(sapply(1:K,function(x){paste0('Cluster',x)})
                                                       %>% unlist,'Loss') 
  write.csv(Weights,file = paste0(output_dir,'variant_Weights.csv'),row.names = FALSE)
  if(K==3){
    ggtern(data = Weights,aes(z = Cluster3,x = Cluster1,y=Cluster2))+
      geom_point(aes(color = Loss)) + scale_fill_viridis(option = "D")+
      scale_color_viridis(option = "D",direction = -1)+
      theme_rgbw() 
    
    ggsave(filename = paste0(output_dir,'variant_ternary.png'),width = 7,height = 7)
  }# if K == 3
  jpeg(paste0(output_dir,'variant_ensemble.png'), width = 480, height = 480)
  superheat(dplyr::select(Weights, -Loss), 
            yr = Weights$Loss,
            yr.axis.name = "Loss",
            yr.plot.type = 'line',
            order.rows = order(Weights$Loss))
  dev.off()  
  
  
  
  
  
  # random ####
  l2 = generate_time(n=n, # simulation size
                     p ,
                     beta_sd,
                     beta_base,
                     random = 'TRUE',
                     seed ,
                     epsilon_sd  )
  source = l2$source; target = l2$target; dat = l2$dat; beta = l2$beta
  write.csv(beta,paste0(output_dir,'invariant_beta.csv'),row.names = FALSE)
  write.csv(dat,paste0(output_dir,'invariant_dat.csv'),row.names = FALSE)
  write.csv(source,paste0(output_dir,'invariant_source.csv'),row.names = FALSE)
  write.csv(target,paste0(output_dir,'invariant_target.csv'),row.names = FALSE)
  source$cluster = cut(source$t,K,labels= 1:K)
  Weights = matrix(NA,nrow=num_simulation,ncol= K+1)
  
  for (i in 1:num_simulation) {
    Weights[i,1:K] = bayesm::rdirichlet(alpha = rep(1,K))
    s = w2c(source,Weights[i,1:K],size = simulation_size)
    model = lm(y~., data = source %>% dplyr::select(-cluster,-t ,-x0) %>%.[s,]) # the second column is time "t"
    Weights[i,K+1] = pred_mse(model,target %>% mutate(y = NULL,t = NULL,x0=NULL),target$y)
  }
  
  Weights = as.data.frame(Weights); names(Weights) = c(sapply(1:K,function(x){paste0('Cluster',x)})
                                                       %>% unlist,'Loss') 
  write.csv(Weights,file = paste0(output_dir,'invariant_Weights.csv'),row.names = FALSE)
  if(K==3){
    ggtern(data = Weights,aes(z = Cluster3,x = Cluster1,y=Cluster2))+
      geom_point(aes(color = Loss)) + scale_fill_viridis(option = "D")+
      scale_color_viridis(option = "D",direction = -1)+
      theme_rgbw() 
    
    ggsave(filename = paste0(output_dir,'invariant_ternary.png'),width = 7,height = 7)
  }# if K == 3
  
  jpeg(paste0(output_dir,'invariant_ensemble.png'), width = 480, height = 480)
  superheat(dplyr::select(Weights, -Loss), 
            
            yr = Weights$Loss,
            yr.axis.name = "Loss",
            yr.plot.type = 'line',
            order.rows = order(Weights$Loss))
  dev.off()  
}




df_adapter = function(p, beta){
  data.frame(Time = 1:nrow(beta),
             beta = beta[,paste0('V',p)])
}

pred_func_time = function(source_in_use,target){
  model = lm(y~.,
             data = source_in_use %>% 
               dplyr::select(-cluster,-t ,-x0) ) # the second column is time "t"
  p = pred_mse(model,target %>% 
                 mutate(y = NULL,t = NULL,x0=NULL),target$y)
}
