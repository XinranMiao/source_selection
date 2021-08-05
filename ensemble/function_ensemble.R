library("factoextra");library(gtools)
# source_split()-------
# input: 
# design matrix
# idx_target, if length == 1, 
# n_cluster(numeric): number of clusters
# seed: random seed
# plotting(logic): whether plot the figure or not
# output_dir

# output:
# data.frame with one column indicating clusters.
source_split = function(df, # data.frame withe the first column indicating predictable variable
                        idx_target = 'random', # the indices of target examples in X, if 'random', then random indices will be drawn
                        target_fraction = NULL, # only required if idx_target == 'random'
                        n_cluster, # number of clusters to split source
                        seed, # random seed
                        val_fraction = 0.5, # The fraction of target data to be validation. The rest are test data.
                        output_dir = '.', # directory of output plots/files
                        plotting = TRUE, # whether or not plot
                        data_type = 'California'
){
  if(idx_target =='random'){
    set.seed(seed)
    idx_target=  sample(1:nrow(df),round(nrow(df) * target_fraction))
  }
  idx_val = sample(1:length(idx_target),round(val_fraction * length(idx_target)))
  clusters = rep(NA,nrow(df))
  clusters[idx_target][idx_val] = 'Validation'
  clusters[idx_target][-idx_val] = 'Test'
  set.seed(seed)
  hc_sparse =df[-idx_target,-1] %>%
    apply(2,scale) %>%
    eclust('kmeans',
           k=n_cluster,
           stand = TRUE,
           graph =FALSE)
  
  clusters[-idx_target] = hc_sparse$cluster  
  
  if(plotting){
    if(data_type == 'California'){
    cbind(df,clusters) %>%ggplot() +
      geom_point(aes(x = Latitude, 
                     y = Longitude, 
                     color = clusters),size = 2) +
      theme_bw()+
      theme(legend.text =element_text(size=14) ,
            legend.title = element_text(size=14,face = 'bold'),
            axis.title = element_text(size=12,face = 'bold'))
    ggsave(paste0(output_dir,'source_split','.png'))
    } # data_type == 'California'
  }# if plotting
  
  Source =df[-idx_target,] %>% 
    mutate(Cluster = clusters[-idx_target])
  Validation = df[idx_target[idx_val],] %>%
    mutate(Cluster = clusters[idx_target][idx_val])
  Test =  df[idx_target[-idx_val],] %>%
    mutate(Cluster = clusters[idx_target][-idx_val])
  return(list(Source = Source,
              Validation = Validation,
              Test = Test))
}# function


# pred_func(): predict
# input ---
# output ---
pred_func = function(source,  val, mod = 'lm'){
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
    
    mse = mean((val$Value - pred)^2) 
    return(mse)
  }# end if
}# end function


# smlt(): Training model based on one set of cluster weights.
# input ---
# n_training (numeric): number of training examples to draw
# p(numeric vector): a vector of cluster weights
# Source (data.frame): Source data, including predictable variables, explanatory variables, and a column of "Cluster"
# Test (data.frame):  Test data, including predictable variables, explanatory variables, and a column of "Cluster"
# Validation (data.frame): Validation data, including predictable variables, explanatory variables, and a column of "Cluster"
# mod (charactor): model type
# output ---
# results: a vector of training loss and cluster weights
# indices: a vector of source indices in use.
smlt = function(n_training = 1000, # number of training examples to draw
                p = c(1/3,1/3,1/3), # a vector of cluster weights
                Source,  # data.frame with a column indicating cluster
                Test,
                Validation = NULL,
                mod = 'lm'
                ){
  indices = list()
  for (id_cluster in 1:(length(p)-1)) {
    indices[[id_cluster]] = sample(which(Source$Cluster==id_cluster),
                                   round(p[1]*n_training), replace = TRUE)
  }
  if(n_training - length(unlist(indices)) >0){
    indices[[length(p)]] = sample(which(Source$Cluster == length(p)),
                                  n_training - length(unlist(indices)),replace = TRUE)
    
  }
  
  indices = unlist(indices)
  temp = rbind(Source[indices,],Validation) %>% select(-Cluster)
  loss = ifelse(is.null(Validation), 
                pred_func(Source[indices,], Test,mod = mod),
                       pred_func(temp, Test,mod = mod))
  
  results = c(loss,p)
  return(list(results = results,indices = indices))
}



# generate_dat()####
# input ---
# Source (data.frame): Source data, including predictable variables, explanatory variables, and a column of "Cluster"
# Test (data.frame):  Test data, including predictable variables, explanatory variables, and a column of "Cluster"
# Validation (data.frame): Validation data, including predictable variables, explanatory variables, and a column of "Cluster"
# n_training: number of training examples.
# n: number of times to simulate
# para(numeric vector): parameter of rdirichlet()
# mod (charactor): model type
# additional_dir(charactor): additional string in the output directory if isn't NULL.
# output---
# dat(data.frame): The first column is loss. The rest columns are weights of different clusters.
generate_dat = function(Source, 
                        Test, 
                        Validation,
                        n_training = 200,
                        n = 1000,# how many time to simulate
                         para = c(1,1,1), # parameter of rdirichlet()
                        mod = 'lm',
                        output_dir = '.',
                        additional_dir = NULL
                        ){
  dat = matrix(ncol = 1+length(para))
  P = rdirichlet(n, para)
  for (i in 1:nrow(P)) {
    (p = P[i,])
    dat = rbind(dat,smlt(n_training = n_training, 
                         p = p,
                           Source = Source,
                           Test = Test,
                           Validation = Validation,
                         mod = mod)$results[1:(1+length(para))]
    )
  }
  
  
  dat = na.omit(dat)
  dat = as.data.frame(dat)
  names(dat)[1] = c('loss')
  if(!is.null(additional_dir)){
    write.csv(dat, paste0(output_dir,'_','dat',additional_dir,'.csv'))
  }
  return(dat)
}



# summary_stat(): compute summary statistic for a vector of weights ####
# input ---
# x: a vector of weights
summary_stat = function(x){
  x = unlist(x)
  u = rep(1 / length(x),length(x))
  return(mean(abs(u - x)))
}


# w2stat(): Compute summary statistic for a data.frame of weights####
# input ---
# Weights: data.frame of weights
# n_cluster: number of clusters
# num_stats: numboer of top combinations to look at.
# output ---
# a numeric value of summary statistics
w2stat = function(Weights, # data.farme
                  n_cluster,
                  num_stats = 5){ # top ? combintations to look at
  apply(Weights[order(Weights[,n_cluster + 1])[1:num_stats],
                1:n_cluster], 
        1, summary_stat) %>% mean
}

# ensemble(): Ensemble plot
# input ---
# dat (data.frame): the first column indicates loss, while the rest are weights of clusters
# output ---
# None. Save the plot if additional_dir isn't NULL.
ensemble = function(dat,
                    output_dir,
                    n_cluster,
                    additional_dir = NULL){
  dat_plot = dat[,c(1,2:ncol(dat))]
  names(dat_plot) = c('Loss',
                      sapply(1:n_cluster,function(x){paste0('Cluster',x)}) %>% 
                        unlist)
  if(! is.null(additional_dir)){
    jpeg(paste0(output_dir,'_ensemble.png'))
  }
  superheat(dplyr::select(dat_plot, -Loss), 
            # scale the variables/columns
            # scale = TRUE,
            
            # add mpg as a scatterplot next to the rows
            yr = dat_plot$Loss,
            yr.axis.name = "Loss",
            # yr.point.size = 4,
            yr.plot.type = 'line',
            order.rows = order(dat_plot$Loss))
  if(! is.null(additional_dir)){
    dev.off()
  }
}
