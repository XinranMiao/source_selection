source('CA_functions.R')
library(stringr);
library(dplyr);library(tidyverse); 
library(ggplot2);library(viridis)

setwd('./output/');system('./prepare.sh')
output_base_dir = '../vis/'
if(!dir.exists(output_base_dir)){ dir.create(output_base_dir)}



# Visualizations about bandit results####
bandit_dfs_dir = read.csv('bandit_results.csv', header = FALSE)[,1] 
bandit_dfs_dir = bandit_dfs_dir[sapply(bandit_dfs_dir,function(x){!str_detect(x,'ex')})]
df = map_dfr(bandit_dfs_dir, ~bandit_df_adapter_ca(.),.id = 'path')

# plot the lines of comparisons for bandit random vs cluster####
output_dir = paste0(output_base_dir,'compare_lines/')
if(!dir.exists(output_dir)){ dir.create(output_dir)}
plot_compare_ca(df,output_dir = output_dir,s = seed)



# plot the stack bar plot ####
output_dir = paste0(output_base_dir,'stack/')
if(!dir.exists(output_dir)){ dir.create(output_dir)}
plot_stack_ca(df,
              output_dir = output_dir,
              s = seed)


# Make ternary plots ####
library(ggtern);library(stringr)
output_dir = paste0(output_base_dir,'ternary/')
if(!dir.exists(output_dir)){ dir.create(output_dir)}
ternary_dirs = read.csv('./ternary_dat.csv', header = FALSE) %>% unlist
for (ternary_dir in ternary_dirs) {
  Weights = read.csv(ternary_dir)[,-1]
  
  names(Weights) = c('Loss','Cluster1','Cluster2','Cluster3')
  ggtern(data = Weights,aes(z = Cluster3,x = Cluster1,y=Cluster2))+
    geom_point(aes(color = Loss)) + scale_fill_viridis(option = "D")+
    scale_color_viridis(option = "D",direction = -1)+
    theme_rgbw() 
  ggsave(paste0(output_dir,'ternary.png'),height = 7,width = 7)
}

