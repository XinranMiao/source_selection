
# Implement the ensemble method and record the corresponding weights.
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

# When K= 3, make a ternary plot
if(K==3){
  ggtern(data = Weights,aes(z = Cluster3,x = Cluster1,y=Cluster2))+
    geom_point(aes(color = Loss)) + scale_fill_viridis(option = "D")+
    scale_color_viridis(option = "D",direction = -1)+
    theme_rgbw() 
  
  ggsave(filename = paste0(output_dir,'variant_ternary.png'),width = 7,height = 7)
}# if K == 3

# Visualize the ensemble result
jpeg(paste0(output_dir,'variant_ensemble.png'), width = 480, height = 480)
superheat(dplyr::select(Weights, -Loss), 
          yr = Weights$Loss,
          yr.axis.name = "Loss",
          yr.plot.type = 'line',
          order.rows = order(Weights$Loss))
dev.off() 
