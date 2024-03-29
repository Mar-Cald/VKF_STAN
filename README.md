# VKF_STAN
An implementation of the Volatile Kalman Filter in STAN. 
VKF is based on this paper: Piray, P., & Daw, N. D. (2020). A simple model for learning in volatile environments. PLoS computational biology, 16(7), e1007963.

#### N.B. In this implementation the initial volatility is fixed.

## Example assuming a response on both type of trials
RTs (shifted-lognormal distribution)
- o.csv : trial list
- vkf_sim_recovery_subj.R : R script to simulate and recover parameters (one subject)
- model_file_sim_subj.stan : simulation (one subject)
- model_file_recovery_subj.stan : recovery (one subj)
- vkf_sim_recovery.R : R script to simulate and recover parameters (multiple subjects)
- model_file_sim.stan : simulation (multiple subjects)
- model_file_recovery.stan : recovery (multiple subj, hierarchical model, non-centered parameterization)

## Example Go-NoGo task
RTs (shifted-lognormal distribution)
(filter Go trials when computing the likelihood)
- model_file_sim_go_nogo.stan : sim just go RT
- model_file_transf.stan : hierachical model (non-centered parameterization) and sigmoid and exp transformation for sigma_v and omega 
- vkf_sim_recovery_transf_go_nogo.R : R script to simulate and recover parameters (multiple subjects), add adapt_delta


### Feel free to add model files/scripts that could be useful for other people (i.e., examples from other tasks, other response models, whatever..)


