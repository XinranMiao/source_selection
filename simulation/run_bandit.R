
num_initial = 20; num_rounds = 30; step = 10
set.seed(seed)
idx_used = sample(1:nrow(source),num_initial,replace = TRUE)
source_in_use = source[idx_used,]
err_initial = pred_func_time(source_in_use,target)


# bandit ####
num_cluster = K
set.seed(seed)
idx_used = sample(1:nrow(source),num_initial,replace = TRUE)
source_in_use = source[idx_used,]
err_initial = pred_func_time(source_in_use,target)



idx = list()
alpha = rep(1,K);beta = rep(1,K)
Clusters = rep(0,num_rounds); errs = rep(0,num_rounds)
for (t in 1:num_rounds) {
  
  pi = c()
  
  for (j in 1:K) {
    set.seed(t+1+seed+j)
    pi[j] = rbeta(1,alpha[j],beta[j])
  }
  set.seed(t+seed)
  Clusters[t] = which.max(pi)
  
  choice = which(source$cluster == Clusters[t])
  set.seed(t+seed)
  idx[[t]] = sample( choice,step,replace = TRUE)
  
  
  source_in_use = rbind(source_in_use, source[idx[[t]],])
  
  errs[t] = pred_func_time(source_in_use,target)
  
  dif = ifelse(t == 1, errs[t] - err_initial,errs[t] - errs[t-1])
  if(dif<0){alpha[Clusters[t]] = alpha[Clusters[t]] + 1
  idx_used = c(idx_used,unlist(idx[[t]]))
  }else{
    beta[Clusters[t]] = beta[Clusters[t]] + 1
    source_in_use = source[idx_used,]
  }
  
  cat('t=',t,',dif = ',round(dif,3),'cluster = ',Clusters[t],'\n')
}


results_selected = data.frame(cluster = Clusters,
                              error = errs)
write.csv(results_selected,paste0(output_dir,'results_selected.csv'),row.names = FALSE)

# random ####
set.seed(seed)
idx_used = sample(1:nrow(source),num_initial,replace = TRUE)
source_in_use = source[idx_used,]
err_initial = pred_func_time(source_in_use,target)

alpha = rep(1,K);beta = rep(1,K)
Clusters = rep(0,num_rounds); errs = rep(0,num_rounds)
for (t in 1:num_rounds) {
  
  set.seed(t-seed)
  Clusters[t] = sample(1:K,1)
  
  choice = which(source$cluster == Clusters[t])
  set.seed(t-seed)
  idx[[t]] = sample( choice,step,replace = TRUE)
  idx_used = c(idx_used,unlist(idx[[t]]))
  
  source_in_use = rbind(source_in_use, source[idx[[t]],])
  
  errs[t] = pred_func_time(source_in_use,target)
  if(dif<0){
  idx_used = c(idx_used,unlist(idx[[t]]))
  }else{
    source_in_use = source[idx_used,]
  }
}


results_random = data.frame(cluster = Clusters,
                            error = errs)
write.csv(results_random,paste0(output_dir,'results_random.csv'),row.names = FALSE)
#jpeg(paste0(output_dir,'bandit_error_random.jpeg'));p_err(errs, Clusters, 
 #                                                         title =paste0('Ultimate summary statistic is ',
     #                                                                   round(summary_stat(table(Clusters) / length(Clusters) ) ,3)),
     #                                                     K);dev.off()
# Plot the difference of errors at each time.
# data.frame(Difference = differences,Cluster = as.factor(Clusters),Rounds = 1:num_rounds) %>% 
#  ggplot() + 
#  geom_point(aes(x = Rounds,y = Difference,color = Cluster)) +
#  geom_line(aes(x = Rounds,y = Difference,color = Cluster)) + 
#  theme_bw()
# ggsave(file = paste0(output_dir,'Difference_in_error.png'))


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
ggsave(paste0(output_dir,'random_vs_bandit.png'),width = 5,height = 5.5)





# If 0 examples are selected from certain cluters, table() function may result in 
# a shorter vector. Therefore, make sure the levels of clusters are correct.
if(length(unique(results_selected$cluster)) != K){
  fct = results_random$cluster %>% as.factor()
  levels(fct) =sapply(1:K,as.character)
  results_selected$cluster = fct
}
if(length(unique(results_random$cluster)) != K){
  fct = results_random$random %>% as.factor()
  levels(fct) =sapply(1:K,as.character)
  results_random$cluster = fct
}
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

ggsave(paste0(output_dir,'paired_bar.png'),height = 5,width = 5)



