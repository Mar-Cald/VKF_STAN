functions {
    // Sigmoid x vkf
  real sigmoid(real x) {
    return 1 / (1 + exp(-x));
  }
}
data {

  // number of rt GO
  int<lower = 1> N_lik;
  // number of observations x subj
  int<lower = 1> N_trial;
  // number of subjects
  int<lower = 1> N_subj;
  int<lower = 1> subj [N_subj*N_trial];

  //stimuli : go vs nogo (i.e, 1 vs 0)
  vector[N_trial * N_subj] GO;
}
parameters {

    // volatile kalman filter parameters
    real<lower = 1e-3, upper = 6> omega;
    vector<lower = -1.5, upper = 1.5>[N_subj] u_om; // subj adjustment
    real<lower = 1e-3, upper = 1-1e-3> sigma_v;
    vector<lower = -1, upper = 1>[N_subj] u_sigma_v; // subj adjustment

    // regression model parameters
    // mu
    real<lower = 4.5, upper = 6.5> intercept;
    real<lower = -1, upper = 1> beta;
    real<lower = -1, upper = 1> beta_nogo;
    vector<lower = -1, upper = 1>[N_subj] u_alpha; // subj adjustment
    vector<lower = -1, upper = 1>[N_subj] u_beta; // subj adjustment

    // ndt
    real<lower = 10, upper = 200> ndt;
    vector<lower = -100, upper = 100>[N_subj] u_ndt; // subj adjustment

    real<lower = 0> sigma;
}
generated quantities {
  
  // predtited rts
  vector[N_lik] rt_pred;
  // for loop
  int tn;
  int i = 1;
  
  // regression model
  real accum1;
  real mu;
  
  // volatile kalman filter (vkf)
  real mpre; // prediction n-1
  real wpre; // variance n-1
  real k; // kalman gain
  real wcov; // covariance
  real o; // input
  real delta_v; // volatility pe

  real v0 = 4; // as estimated by ideal obs
  
  real v = v0;  // Initialize volatilities
  real w = omega + u_om[subj[1]]; // Initialize posterior variances
  real m = 0; // Initialize predictions

  vector[N_trial * N_subj] predictions;
  vector[N_trial *  N_subj] volatility;
  vector[N_trial * N_subj] delta_m; //pred_error

  for (n in 1:N_subj*N_trial) {
    
        o = GO[n]; // input
        predictions[n] = m; 
        volatility[n] = v; 
      
        mpre = m; // p_prediction
        wpre = w; // p_variance
      
        delta_m[n] = o - sigmoid(mpre); // prediction error

        k = (wpre + v) / (wpre + v + (omega + u_om[subj[n]])); // Kalman Gain/learning rate
        
        m = mpre + sqrt(wpre + v) * delta_m[n];
      
        w = (1 - k) * (wpre + v);  // variance update
        wcov = (1 - k) * wpre; // covariance
        delta_v     = (m-mpre)^2 + w + wpre - 2*wcov - v; 

        v   = v + (sigma_v+ u_sigma_v[subj[n]]) * delta_v;
      

      //shifted lognormal
      // regression model
      if (GO[n] == 1){
      mu = intercept + u_alpha[subj[n]] + sigmoid(predictions[n]) * (beta + u_beta[subj[n]]);
      accum1 = lognormal_rng(mu, sigma);
      rt_pred[i] = accum1 + ndt + u_ndt[subj[n]]; // deicision time = rt - non-decision time
      i = i + 1;
      }
      // loop across subj
      tn = n < (N_trial * N_subj) ? n : n-1;
      v = subj[n] != subj[tn+1] ? v0 : v; // Initialize volatilities
      w = subj[n] != subj[tn+1] ? omega + u_om[subj[tn+1]] : w;  // Initialize posterior variances
      m = subj[n] != subj[tn+1] ? 0 : m; // Initialize predictions
      
      }
}

