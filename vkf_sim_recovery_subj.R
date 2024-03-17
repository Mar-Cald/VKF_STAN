# -----------------------------------------------------
# Simulation VKF + Parameter Recovery One Subject
# -----------------------------------------------------

rm(list = ls())

set.seed(15595)
options(scipen = 20)

# Import libraries ---------------------
ipak <- function(pkg) {
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = T)
  sapply(pkg, require, character.only = T)
}
packages <- c("rstan", "rstansim", "readr", "MASS", "bayesplot") 
ipak(packages)


# Simulation VKF  1 subj ---------------
o <- read_csv("o.csv") # Trial list
n_subj <- 1 #set number of subjects
n_trial<- 448 # number of trials/subj
go_dat <- rep(o$GO,n_subj) # repeat trial list for each subject

## data 
dat_mod <- list(N_trial = n_trial,
                N_subj = n_subj,
                subj = rep(1:n_subj, each = n_trial),
                GO = go_dat)


simulation <- simulate_data(
  file = "mod_file_sim_1subj.stan", 
  input_data = dat_mod,
  param_values = list("sigma" = 0.3,
                      "intercept" = 4.8,
                      "beta" = -0.3,
                      "ndt" = 150,
                      "omega" = 0.5,
                      "sigma_v" = 0.1), 
  data_name = "sim_data_sigma_v",
  nsim = 1, # number of sim
  vars = c("predictions","volatility","rt_pred")
)

# load simulated data
dat_sim <- readRDS("sim_data_1.rds")

# Recovery -----------------------------
dat_mod <- list(N_trial = dat_mod$N_trial,
                N_subj = dat_mod$N_subj,
                subj = dat_mod$subj,
                GO = dat_mod$GO,
                rt = dat_sim$rt_pred,
                minRT = min(dat_sim$rt_pred))

## sampling init
initfun <- function() {
  list(
    ndt = runif(1,60,80),
    omega = runif(1,0.2,2),
    sigma_v = runif(1,0.1,0.5)
  )
}

##  run model
recovery <- stan(
  file = "model_file_recovery.stan",# Stan program
  data = dat_mod, chains = 2, cores = 2,
  iter = 8000, init = initfun)

df <- as.data.frame(recovery)
mcmc_recover_hist(
  x = df[1:6],
  true = c(0.5,0.2,0.3,5.3,-0.3,150) # set true values
)

## divergent, correlations, posterior
np_ncp <- nuts_params(recovery)
mcmc_pairs(recovery,pars = c("intercept","ndt","beta",
                             "omega","sigma_v"),
           np = np_ncp)

#------------------------------------------------------------