# VKF_STAN
An implementation of the Volatile Kalman Filter in STAN. Work in progress...

VKF is based on this paper: Piray, P., & Daw, N. D. (2020). A simple model for learning in volatile environments. PLoS computational biology, 16(7), e1007963.

## Example assuming a response on both type of trials
RTs (shifted-lognormal distribution), v0 fixed
- o.csv : trial list
- vkf_sim_recovery_subj.R : R script to simulate and recover parameters (one subject)
- model_file_sim_subj.stan : simulation (one subject)
- model_file_recovery_subj.stan : recovery (one subj)
- vkf_sim_recovery.R : R script to simulate and recover parameters (multiple subjects)
- model_file_sim.stan : simulation (multiple subjects)
- model_file_recovery.stan : recovery (multiple subj, hierarchical model, non-centered parameterization)

## Example Go-NoGo task 
RTs (shifted-lognormal distribution) + Bernoulli
.....


### If you notice errors please let me know. 
Feel free to add model files/scripts that could be useful for other people (i.e., examples from other tasks, other response models, whatever..)


