source('CA_ensemble_functions.R')
source('CA_bandit_functions.R')
library("readr");library(dplyr);
library(superheat);library(plyr);library(viridis);library(ggplot2)


# Import dataset ####
cnames <- c("Value","Income","Age", "Room","Bedroom","Population", "Households","Latitude","Longitude")
housing <- read_table("cadata.txt", col_names = cnames) %>%
  mutate(
    AvgRooms = Room / Households,
    AvgBedrooms = Bedroom / Households,
    Value = Value / 1000
  ) %>%
  select(-Room, -Bedroom)

base_dir = './output/'
if(!dir.exists(base_dir)){dir.create(base_dir)}

num_rounds = 200 
step = 50 
num_initial = 30
n_val = 10
seed = 9







for(K in 2:5){
  seed = seed; K = K
  cat('seed = ',seed,', K = ',K,'\n')
  output_dir = paste0(base_dir,'seed',seed,'_K',K,'/','seed',seed,'_K',K)
  if(!dir.exists(paste0(base_dir,'seed',seed,'_K',K,'/'))){
    dir.create(paste0(base_dir,'seed',seed,'_K',K,'/'))
  }
  
  idx_target =choose_target(seed = seed, 
                            housing)
  
  length(idx_target)
  l = source_split(df = housing, 
                   idx_target = idx_target,
                   target_fraction = NULL,
                   n_cluster = K,
                   seed = seed, 
                   val_fraction = 0.1,
                   output_dir = output_dir, 
                   plotting = TRUE, 
                   data_type = 'California'
  )
  
  
  Source = l$Source; Validation = l$Validation; Test = l$Test
  write.csv(Source,paste0(output_dir,'Source.csv'),row.names = FALSE)
  write.csv(Validation,paste0(output_dir,'Validation.csv'),row.names = FALSE)
  write.csv(Test,paste0(output_dir,'Test.csv'),row.names = FALSE)
  Test = Test[sample(1:nrow(Test),n_val),]
  
  
  
  # ensemble ####
  dat = generate_dat(Source, Test, Validation,
                     n_training = 1000, # 1000
                     n = 600, # 600
                     para =rep(1,K),
                     mod = 'loglm',
                     output_dir = output_dir,
                     additional_dir = '')   
  
  
  
  # Plot the summary statistic -----
  s = rep(1,100)
  for (i in 1:100) {
    s[i] = w2stat(dat[,c(2:ncol(dat),1)], n_cluster = K,num_stats = i)
  }
  jpeg(paste0(output_dir,'_w2stat.jpeg'));plot(1:100,s,xlab = 'Number of Weight Combinations',ylab = 'Summary Statistic',main = paste0(K,' clusters'));lines(1:100,s);dev.off()
  
  # Ensemble plot
  ensemble(dat,output_dir = output_dir,
           n_cluster = K,additional_dir = '')
  # bandit ####
  results_selected = sample_selection(Source, Validation, Test, 
                                      seed,
                                      K,
                                      split_criteria  = 'cluster',
                                      output_dir,
                                      num_initial = num_initial,
                                      step = step,
                                      num_rounds = num_rounds)
  prediction = results_selected$predictions
  results_selected = results_selected$results
  results_random = sample_selection(Source, Validation, Test, 
                                    seed,
                                    K,
                                    split_criteria  = 'random',
                                    output_dir,
                                    num_initial = num_initial,
                                    step = step,
                                    num_rounds = num_rounds)
  
  results_random = results_random$results
  
  random_vs_bandit_vis(results_random, results_selected, output_dir)
  
} #K



