functions {
  // Sigmoid x vkf
  real sigmoid(real x) {
    return 1 / (1 + exp(-x));
  }
}
data {
  
  // number of observations 
  int<lower = 1> N_trial;
  //stimuli : go vs nogo (i.e., 1 vs 0)
  vector[N_trial] GO;
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
  real<lower = 10, upper = 300> ndt;
  
  real<lower = 0> sigma;
}
generated quantities {
  
  vector[N_trial] rt_pred;

  // regression model
  real accum1;
  real mu;
  
  // volatile kalman filter 
  real mpre; // prediction n-1
  real wpre; // variance n-1
  real k; // kalman gain
  real wcov; // covariance
  real o; // input
  real delta_v; 
  
  // init
  real v0 = 4; // Initial volatility
  real v = v0;  // Initialize volatilities
  real w = omega; // Initialize posterior variances
  real m = 0; // Initialize predictions
  
  vector[N_trial] predictions;
  vector[N_trial] volatility;
  vector[N_trial] delta_m; //pred_error

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
    delta_v     = (m-mpre)^2 + w + wpre - 2*wcov - v; // volatility pe
    
    v   = v + sigma_v * delta_v; // volatility update
    
    // regression model
    //shifted lognormal
    mu = intercept + predictions[n] * GO[n] * beta;
    accum1 = lognormal_rng(mu, sigma);
    rt_pred[n] = accum1 + ndt; // add non-decision time
    
  }
}


