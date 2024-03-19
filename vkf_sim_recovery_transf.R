# -----------------------------------------------------
# Simulation VKF + Parameter Recovery Multiple Subjects 
# sigmoid trans on sigma_v and exp on omega
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


# Simulation VKF multiple subjects ---------------
o <- read_csv("o.csv") # Trial list
n_subj <- 30 #set number of subjects
n_trial<- 448 # number of trials/subj
go_dat <- rep(o$GO,n_subj) # repeat trial list for each subject

## data 
dat_mod <- list(N_trial = n_trial,
                N_subj = n_subj,
                subj = rep(1:n_subj, each = n_trial),
                GO = go_dat)

## Non-centered parametrization for individual parameters
### Intercept, beta, ndt, omega, sigma_v
N_adj <- 5 # number of adjustments
tau_u <- c(rep(.1, N_adj-2),0.1,10) # small between subj variability
rho <- .2 # corr
Cor_u <- matrix(rep(rho, N_adj * N_adj), nrow = N_adj)
diag(Cor_u) <- 1
Sigma_u <- diag(tau_u, N_adj, N_adj) %*%
  Cor_u %*%
  diag(tau_u, N_adj, N_adj)
u <- mvrnorm(n = n_subj, rep(0, N_adj), Sigma_u)
#max(u[,1]);max(u[,2]);max(u[,3]);max(u[,4]);max(u[,5]); # check 
#min(u[,1]);min(u[,2]);min(u[,3]);min(u[,4]);min(u[,5]); # check 

simulation <- simulate_data(
  file = "model_file_sim.stan", 
  input_data = dat_mod,
  param_values = list("sigma" = 0.3,  # set true values
                      "intercept" = 5.3,
                      "u_alpha" = u[,1],
                      "beta" = -0.3,
                      "u_beta" = u[,2],
                      "ndt" = 150,
                      "u_ndt" = u[,5],
                      "omega" = 0.5,
                      "u_om" = u[,3],
                      "sigma_v" = 0.03,
                      "u_sigma_v" = u[,4]), 
  data_name = "sim_data",
  nsim = 1, # set number of simulation
  vars = c("rt_pred") # set 
)

# load simulated data
dat_sim <- readRDS("sim_data_1.rds")

# Recovery -----------------------------
dat_mod <- list(N_trial = dat_mod$N_trial,
                N_subj = dat_mod$N_subj,
                subj = dat_mod$subj,
                GO = dat_mod$GO,
                rt = ifelse(dat_mod$GO == 1, dat_sim$rt_pred, 99999), #NoGo to 999999
                minRT = min(dat_sim$rt_pred),
                N_lik = sum(dat_mod$GO == 1 & dat_sim$rt_pred != 99999), # lik on Go trials only, 
                N_re = 5) #number of adjustments

## sampling init
initfun <- function() {
  list(
    #omega = runif(1,0.2,2),
    #sigma_v = runif(1,0.1,0.5),
    z_u = matrix(rnorm(N_adj*n_subj, 0, 0.1),
                 N_adj, n_subj),
    L_u = diag(N_adj)
  )
}

##  run model
recovery <- stan(
  file = "model_file_transf.stan",# Stan program
  data = dat_mod, chains = 4, cores = 4,
  iter = 10000, init = initfun, control = list(adapt_delta = .95))

save(recovery, "recovery_transf.rda")

df <- as.data.frame(recovery)
mcmc_recover_hist(
  x = df[1:6],
  true = c(0.5,0.2,0.3,5.3,-0.3,150) # set true values (check)
)

## divergent, correlations, posterior
np_ncp <- nuts_params(recovery)
mcmc_pairs(recovery,pars = c("intercept","ndt","beta",
                             "omega","sigma_v"),
           np = np_ncp)

#------------------------------------------------------------
