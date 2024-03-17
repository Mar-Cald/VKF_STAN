functions {
    // Sigmoid x vkf
  real sigmoid(real x) {
    return 1 / (1 + exp(-x));
  }
}
data {

  // number of observations
  int<lower = 1> N_trial;

  //stimuli : go vs nogo
  vector[N_trial] GO;

  // response
  vector[N_trial] rt;

  //min reaction time, for non decision time
  real<lower=1> minRT;
}
parameters {

    // volatile kalman filter parameters
    real<lower = 1e-3, upper = 6> omega;
    real<lower = 1e-3, upper = 1-1e-3> sigma_v;

    // regression model parameters
    // mu
    real<lower = 4.5, upper = 6.5> intercept;
    real<lower = -1, upper = 1> beta;

    // ndt
    real<lower = 10, upper = minRT> ndt;

    real<lower = 0> sigma;
}
model {
  
  vector[N_trial] log_lik;
  // regression model
  real T;
  real mu;
  
  // volatile kalman filter (vkf)
  real mpre; // prediction n-1
  real wpre; // variance n-1
  real k; // kalman gain
  real wcov; // covariance
  real o; // input
  real delta_v; // volatilty pe

  real v0 = 4; // as estimated by ideal obs
  real v = v0;  // Initialize volatilities
  real w = omega; // Initialize posterior variances
  real m = 0; // Initialize predictions

  vector[N_trial] predictions;
  vector[N_trial] volatility;
  vector[N_trial] delta_m; //pred_error
  
  // regression model
   target += normal_lpdf(sigma | .3, .2)
    - normal_lccdf(0 | .3, .2);
  // mu
  target += normal_lpdf(intercept |4.8, 1.5);
  target += normal_lpdf(beta | -0.3, 0.5);

  // ndt
  target += normal_lpdf(ndt | 150, 100)
    - log_diff_exp(normal_lcdf(minRT | 150, 100),  // upper minRT
                   normal_lcdf(10 | 150, 100));  // lower 10

  // vkf model priors
  target += normal_lpdf(omega | 3, 1.5)
       - log_diff_exp(normal_lcdf(6 | 3, 1.5),  // upper minRT
                 normal_lcdf(1e-3 | 3, 1.5));  // lower 10
                 
  target += normal_lpdf(sigma_v | 0.9, 0.5)
       - log_diff_exp(normal_lcdf(1-1e-3 | 0.9, 0.5),  // upper minRT
                 normal_lcdf(1e-3 | 0.9, 0.5));  // lower 10


  for (n in 1:N_trial) {

        o = GO[n]; // input
        predictions[n] = m; 
        volatility[n] = v; 
        mpre = m; // p_prediction
        wpre = w; // p_variance
        delta_m[n] = o - sigmoid(mpre); // prediction error
        k = (wpre + v) / (wpre + v + omega); // Kalman Gain/learning rate
        m = mpre + sqrt(wpre + v) * delta_m[n];
        w = (1 - k) * (wpre + v);  // variance update
        wcov = (1 - k) * wpre; // covariance
        delta_v     = (m-mpre)^2 + w + wpre - 2*wcov - v; 
        v   = v + sigma_v * delta_v;
      
      // regression model
      T = rt[n] - ndt ;  // deicision time = rt - non-decision time
      mu = intercept + predictions[n] * GO[n] * beta;
    
      //shifted lognormal
      log_lik[n] = lognormal_lpdf(T | mu, sigma); 
      }
  target += sum(log_lik);
}


