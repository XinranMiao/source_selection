setwd('/Users/miaoxinran/Documents/projects/transfer/codes/simulation')
source('simulation_functions.R')
output_dir_base = './output/'
if(!dir.exists(output_dir_base)){
  dir.create(output_dir_base)
}
K=3; p=4; seed =1
n = 1000; num_simulation = 1000; epsilon_sd = 0.1

output_dir_base = paste0(output_dir_base,'K',K,'p',p,'/')
if(!dir.exists(output_dir_base)){
  dir.create(output_dir_base)
}


# --------------------------------------------------------------------
# Case 1: the relationship between y and x is affected by time ####

output_dir = paste0(output_dir_base,'variant_')
beta_base = c(.9,.2,-.3,.3)
beta_sd = c(.01,.1,.04,.1)

# Generate data
l = generate_time(n, # simulation size
                  p ,
                  beta_sd = beta_sd,
                  beta_base =beta_base,
                  random = FALSE,
                  continuous = FALSE,
                  seed ,
                  epsilon_sd =0.1 )
source = l$source; target = l$target; dat = l$dat;beta = l$beta;B=l$B
source$cluster = cut(source$t,K,labels= 1:K)

# Ensemble method
source('run_ensemble.R')


# bandit selection
num_initial = 20; H = 30; b = 10
source('run_bandit.R')


# --------------------------------------------------------------------
# Case 2: the relationship between y and x isn't affected by time ####

output_dir = paste0(output_dir_base,'invariant_')

# Generate data
l = generate_time(n, # simulation size
                  p ,
                  beta_sd = beta_sd,
                  beta_base =beta_base,
                  random = TRUE,
                  continuous = TRUE,
                  seed ,
                  epsilon_sd = epsilon_sd)
source = l$source; target = l$target; dat = l$dat;beta = l$beta
source$cluster = cut(source$t,K,labels= 1:K)

source('run_ensemble.R')
source('run_bandit.R')
