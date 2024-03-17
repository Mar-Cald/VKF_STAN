functions {
  // Sigmoid x vkf
  real sigmoid(real x) {
    return 1 / (1 + exp(-x));
  }
}
data {
  
  // number of observations x subj
  int<lower = 1> N_trial;
  // number of participants
  int<lower = 1> N_subj;
  int<lower = 1> subj [N_subj*N_trial];
  
  //stimuli : go vs nogo (i.e 1 vs 0)
  vector[N_trial * N_subj] GO;

  // response
  vector[N_trial*N_subj] rt;
  
  //min reaction time, for non decision time
  real minRT;  // upper bound 
  
  // number of individual adj
  int<lower = 1> N_re;
}
parameters {
  
  // volatile kalman filter parameters
  real<lower = 1e-3, upper = 4> omega;
  real<lower = 1e-3, upper = 1-1e-3> sigma_v;

  // regression model parameters
  // sigma
  real<lower = 0> sigma;
  // mu
  // Group
  real<lower = 4, upper = 6.6> intercept;
  real<lower = -1.5, upper = 1.5> beta;
  real<lower = 50, upper = minRT> ndt;
  
  // Individual 
  vector<lower = 0>[N_re] tau_u;
  matrix[N_re, N_subj] z_u;
  cholesky_factor_corr[N_re] L_u;
  
}
transformed parameters {
  // regression model non-center
  matrix[N_subj, N_re] u;
  u = (diag_pre_multiply(tau_u, L_u) * z_u)';
}
model {
  
  // loop
  real log_lik[N_trial * N_subj];
  int tn;

  // regression model
  real T;
  real mu;
  
  
  // volatile kalman filter (vkf)
  real mpre; // prediction n-1
  real wpre; // variance n-1
  real k; // kalman gain
  real wcov; // covariance
  real o; // input
  real delta_v; 
  
  real v0 = 4; // Initial volatility
  
  real v = v0;  // Initialize volatilities
  real w = omega; // Initialize posterior variances
  real m = 0; // Initialize predictions
  
  vector[N_trial * N_subj] predictions;
  vector[N_trial *  N_subj] volatility;
  vector[N_trial * N_subj] delta_m; //pred_error
  vector[N_trial * N_subj] vari; //pred_error
  
  
  // Priors
  
  // regression model
  target += normal_lpdf(sigma | .3, .2)
    - normal_lccdf(0 | .3, .2);
  
  // mu
  target += normal_lpdf(intercept | 5.3, 1);
  target += normal_lpdf(beta | -0.3, 0.2);
  target += normal_lpdf(tau_u[1:4] | .1, .1)
    - 4 * normal_lccdf(0 | .1, .1);
  target += normal_lpdf(tau_u[5] | 10, 10)  // ndt
    -  normal_lccdf(0 | 10, 10);
  target += lkj_corr_cholesky_lpdf(L_u | 4);
  target += std_normal_lpdf(to_vector(z_u));

  // ndt
  target += normal_lpdf(ndt | 150, 100)
   - log_diff_exp(normal_lpdf(minRT| 150, 100),  
                     normal_lpdf(50| 150, 100));

  // vkf model priors
  // omega
  target += normal_lpdf(omega | 0.5, 1)
       - log_diff_exp(normal_lcdf(6 | 0.5, 1),  
                 normal_lcdf(1e-3 | 0.5, 1));  
  // sigma_v : volatility learning rate
  target += normal_lpdf(sigma_v | 0.1, 1)
       - log_diff_exp(normal_lcdf(1-1e-3| 0.1, 1),  
                 normal_lcdf(1e-3 | 0.1, 1));  

  for (n in 1:N_subj*N_trial) {
    
        o = GO[n]; // input
        predictions[n] = m; 
        volatility[n] = v; 
      
        mpre = m; // p_prediction
        wpre = w; // p_variance
      
        delta_m[n] = o - sigmoid(mpre); // prediction error
        k = (wpre + v) / (wpre + v + (omega + u[subj[n], 3])); // Kalman Gain/learning rate
        m = mpre + sqrt(wpre + v) * delta_m[n]; // update
        w = (1 - k) * (wpre + v);  // variance update
        wcov = (1 - k) * wpre; // covariance
        delta_v     = (m-mpre)^2 + w + wpre - 2*wcov - v;  // volatility per
        v   = v + (sigma_v + u[subj[n], 4]) * delta_v; // volatility update
      
    // regression model
      T = rt[n] - ndt + u[subj[n], 5];  // deicision time = rt - non-decision time
      mu = intercept + u[subj[n], 1] + predictions[n] * GO[n] * (beta + u[subj[n],2]);
    //shifted lognormal
    log_lik[n] = lognormal_lpdf(T | mu, sigma); 
    
    // Initialize if subj !=
    tn = n < (N_trial * N_subj) ? n : n-1;
  
    v = subj[n] != subj[tn+1] ? v0 : v; // Initialize volatilities
    w = subj[n] != subj[tn+1] ? omega : w;  // Initialize posterior variances
    m = subj[n] != subj[tn+1] ? 0 : m; // Initialize predictions

      }
  target += sum(log_lik);
}
