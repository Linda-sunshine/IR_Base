package markovmodel;

import java.util.Arrays;

import utils.Utils;

public class FastRestrictedHMM {

	private int number_of_topic;
	private int number_of_states;
	private double norm_factor[];
	private double alpha [][];
	private double beta[][];
	
	public FastRestrictedHMM()
	{
		number_of_topic = 0;
		number_of_states = 0;
	}
	
	public double ForwardBackward(double epsilon, double[] theta, double [][] local, double [] pi, double[][] sprobs)
	{
		double loglik;
		this.number_of_topic = theta.length;
		this.number_of_states = 2*this.number_of_topic;
		this.norm_factor  = new double [local.length];
		//double dummy [] = new double [this.number_of_states];
		
		
		alpha  = new double [local.length][this.number_of_states];
		beta = new double [local.length][this.number_of_states];
		
		InitAlpha(pi, local[0],alpha[0]);
		ComputeAllAlphas(local, theta, epsilon, alpha);
		InitBeta(norm_factor[local.length-1], beta[local.length-1]);
		ComputeAllBetas(local, theta, epsilon, beta);
		CombineAllProbs(alpha, beta, sprobs);
		loglik = ComputeLoglik();
		
		
		return loglik;
	}
	
	public void InitAlpha(double[] pi, double[]local0, double[] alpha0) 
	{
		this.norm_factor[0] = 0;
		for (int i = 0; i < this.number_of_topic; i++) 
		{
			alpha0[i] = local0[i]*pi[i];
			alpha0[i+this.number_of_topic] = local0[i]*pi[i+this.number_of_topic];
			this.norm_factor[0] += alpha0[i] + alpha0[i+this.number_of_topic];
		}
		
		Utils.scaleArray(alpha0, 1.0/this.norm_factor[0]);//Hongning: please use the shared implementation
	}
	
	// This method initializes beta[T-1] to be all ones.
	public void InitBeta(double norm, double[] beta_T_1) {
//	  for (int i = 0; i < this.number_of_states; i++) {
//	    beta_T_1[i] = 1;
//	  }
	  Arrays.fill(beta_T_1, 1.0/norm);//Hongning: is this equivalent?
	}
	
	
	public void ComputeAllAlphas(double[][] local, double[] theta, double epsilon, double[][] alpha) 
	{
		for (int i = 1; i < local.length; i++) 
		{
			ComputeSingleAlpha(local[i], theta, epsilon, i, alpha[i-1], alpha[i]);
		}
	}
	
	public void ComputeSingleAlpha(double[] local_t, double[] theta, double epsilon, int norm_index, double[] alpha_t_1, double[] alpha_t) 
	{
		norm_factor[norm_index] = 0.0;
		for (int s = 0; s < this.number_of_topic; s++) {
			alpha_t[s] = epsilon*theta[s]*local_t[s];  // regardless of the previous
			// topic - remember that sum_k alpha[t-1][k] is 1 (because of the norm).
			alpha_t[s+this.number_of_topic] = (1-epsilon)*(alpha_t_1[s] + alpha_t_1[s+this.number_of_topic])*local_t[s];
			norm_factor[norm_index] += alpha_t[s]+alpha_t[s+this.number_of_topic];
		}
		
		Utils.scaleArray(alpha_t, 1.0/norm_factor[norm_index]);//Hongning: please use the shared implementation
	}
	
	
	public void ComputeAllBetas(double[][] local, double[] theta, double epsilon, double[][] beta)  {
		for (int i = local.length - 2; i >= 0; i--) {
			ComputeSingleBeta(local[i+1], theta, epsilon, norm_factor[i], beta[i+1], beta[i]);
		}
	}
	
	
	// This method computes the betas for a single level after beta has been
	// computed for the next level.
	public void ComputeSingleBeta(double[] local_t_1, double[] theta, double epsilon, double norm, double[] beta_t1, double[] beta_t) {
	  double trans_sum = 0;
	  
	  for (int i = 0; i < this.number_of_topic; i++) {
	    trans_sum += epsilon*theta[i]*local_t_1[i]*beta_t1[i];
	  }

	  for (int s = 0; s < this.number_of_topic; s++) {
	    // Recall that beta_t1[s] == beta_t1[s+topics_]
	    beta_t[s] = trans_sum + (1-epsilon)*local_t_1[s]*beta_t1[s];
	    beta_t[s] /= norm;
	    beta_t[s+this.number_of_topic] = beta_t[s];
	  }
	  // we've already normalized the betas!
	}
	
	
	public void CombineAllProbs(double[][] alpha, double[][] beta, double[][] sprobs)
    {
		
		//System.out.println("Alpha:"+alpha.length+"Beta: "+beta.length+"spros: "+sprobs.length);
		for (int i = 0; i < alpha.length; i++) {
			CombineSingleProb(alpha[i], beta[i], sprobs[i]);
		}
    }
	
	
	// This method combines the alpha and beta to get probabilities for a
	// single level.
	public void CombineSingleProb(double[] alpha, double[] beta, double[] sprobs) {
	  double norm = 0;
	  for (int s = 0; s < this.number_of_states; s++) {
	    sprobs[s] = alpha[s]*beta[s];
	    norm += sprobs[s];
	  }
	  Utils.scaleArray(sprobs, norm);//Hongning: please use the shared implementation
	}
	
	
	public double ComputeLoglik() 
	{
		double loglik = 0.0;
		for (int t = 0; t < norm_factor.length; t++) {
			loglik += Math.log(norm_factor[t]);
		}
		
		return loglik;
	}
	
	
}
