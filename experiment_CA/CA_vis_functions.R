library(stringr);
library(dplyr);library(tidyverse); 
library(ggplot2);library(viridis);library(ggpubr)
# extract information from the directory
extract_info_ca = function(directory 
){
  bandit_df = read.csv(directory)
  file_name = substr(directory, str_locate(directory,'/')[2]+1,nchar(directory) ) 
  K = substr(file_name, str_locate(file_name,'K')[2]+1,str_locate(file_name,'K')[2]+1) %>% as.numeric()
  seed = substr(directory, str_locate(file_name,'seed')[2]+1,str_locate(file_name,'K')[2]-2 )  %>% as.numeric()
  split_criteria = substr(file_name, 
                          str_locate(file_name,'results')[2]+2,
                          str_locate(file_name,'.csv')[1]-1 ) 
  
  return(data.frame(K = K,
                    split_criteria = split_criteria,
                    seed = seed))
}   
cumulative_weight_ca = function(bandit_df, 
                                h,
                                k){
  w = bandit_df$cluster[1:h] %>% 
    table %>% 
    .[as.character(k)]
  if(is.na(w)==TRUE){w = 0}else{w = w / h}
  return(w) # weight
}


cumulative_stat_ca = function(bandit_df_dir, # the directory of bandit dataframe,./0926_results_target0_K10_seed0_cluster_DeepResnetPca/bandit_T_sample_1.csv
                              h){
  K = extract_info_ca(bandit_df_dir)$K
  bandit_df = read.csv(bandit_df_dir)
  statistic = abs(table(bandit_df$cluster[1:h]) / h - 1/K ) #|wk - 1/k| 
  statistic = sum(statistic ) / K
  return(statistic)
}

bandit_df_adapter_ca = function(bandit_df_dir){
  bandit_df = read.csv(bandit_df_dir)
  info = extract_info_ca(bandit_df_dir)
  H = nrow(bandit_df) # number of iterations
  K = info$K
  output_df = data.frame(K = info$K,
                         seed = info$seed,
                         split_criteria = info$split_criteria,
                         Error = rep(bandit_df$error,each = K),
                         h = rep(1:H,each = K),
                         k = rep(1:K, H))
  output_df = output_df %>% mutate(
    weight = apply(output_df,1,function(x){
      h = as.numeric(x['h'])%>% unlist
      k = x['k'] %>% as.numeric
      return(cumulative_weight_ca(bandit_df,h,k = k))
    }),
    
    statistic = apply(output_df,1,function(x){
      h = as.numeric(x['h'])%>% unlist
      return(cumulative_stat_ca(bandit_df_dir,h))
    }) 
  ) # mutate
  
  return(output_df)
}


plot_compare_ca = function(df, # of a specific target
                           output_dir,
                           s = NA){
  #df$split_criteria[df$split_criteria=='selected']='bandit'
  #df$split_criteria = factor(df$split_criteria, levels = c('bandit','random'))
  if(is.na(s)==FALSE){
    df = df%>% filter(seed == s)
  }
  # line plot ####
  df %>% 
    ggplot() + 
    geom_line(aes(x = h, y = Error, color = split_criteria,
                  #linetype = split_criteria
    ),size = 0.7) + 
    facet_grid(seed ~ K) +
    theme_bw() +
    xlab('Iterations') + 
    ylab('Error') + 
    theme(axis.text = element_text(size = 9),
          axis.title = element_text(face="bold",size=14)) + 
    scale_color_manual(values =c('#F8766D', '#00BFC4', '#C77CFF'))
  
  ggsave(paste0(output_dir,
                'lines.png'),width = 12,height = ifelse(is.na(s),9,4))
}


plot_stack_ca = function(df, # of a specific target
                         output_dir,
                         s = NA){
  if(is.na(s)==FALSE){
    df = df%>% filter(seed == s)
  }
  df %>%
    #filter(seed==0) %>%
    mutate(k = as.factor(k)) %>%
    ggplot(aes(x = h,y = weight,fill = k, color = k))  +
    geom_bar(position="fill", stat="identity")  +
    facet_grid( split_criteria ~ K)  + 
    geom_line(aes(x = h, y = statistic + 1),
              color = "gray50"
              #size = 1
    ) +
    scale_y_continuous(
      # Features of the first axis
      name = "Weight",
      # Add a second axis and specify its features
      sec.axis = sec_axis( trans=~(. -1), name="Summary Statistic"),
      expand = c(0,0)
    ) +
    scale_x_continuous(expand = c(0,0)) + 
    theme_bw() 
  ggsave(paste0(output_dir,
                'stack.png'),width = 9,height = ifelse(is.na(seed),9,4))
}



# individual point facet
individual_point = function(Ke,
                            df,
                            ...){
  df %>% 
    filter(K == Ke)  %>%
    ggplot() + 
    geom_point(aes(x = h,y =k,size = weight,color = statistic))+
    #facet_grid(seed ~ .) +
    theme_bw() +
    xlab('Iterations') + 
    ylab('Source subsets')+
    theme(...) + 
    scale_size(range = c(0.5,4)) + 
    scale_color_viridis(option = "D",direction = 1)+ 
    scale_y_continuous(breaks=1:Ke) 
}




# To plot the variable densities across clusters
ca_df_adapter = function(dat,variable){
  if(variable %in% c('Population','Households','AvgBedrooms', 
                     'AvgRooms','Income','Value')){
    dat[,variable] = log(dat[,variable])
  }
  output_df = data.frame(Variable = variable,
                         Index = 1:nrow(dat),
                         Observation = (dat[,variable] - mean(dat[,variable]))/sd(dat[,variable]),
                         Cluster = dat$Cluster)
  
  return(output_df)
}

