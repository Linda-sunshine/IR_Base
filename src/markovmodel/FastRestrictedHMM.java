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
		
		alpha  = new double [local.length][this.number_of_states];
		beta = new double [local.length][this.number_of_states];
		
		InitAlpha(pi, local[0]);
		ComputeAllAlphas(local, theta, epsilon);
		InitBeta(norm_factor[local.length-1],local.length-1);
		ComputeAllBetas(local, theta, epsilon);
		CombineAllProbs(sprobs);
		loglik = ComputeLoglik();
		
		return loglik;
	}
	
	public void InitAlpha(double[] pi, double[]local0) 
	{
		this.norm_factor[0] = 0;
		for (int i = 0; i < this.number_of_topic; i++) 
		{
			this.alpha[0][i] = local0[i]*pi[i];
			this.alpha[0][i+this.number_of_topic] = local0[i]*pi[i+this.number_of_topic];
			this.norm_factor[0] += this.alpha[0][i] + this.alpha[0][i+this.number_of_topic];
		}
		Utils.scaleArray(this.alpha[0], 1.0/this.norm_factor[0]);
	}
	
	// This method initializes beta[T-1] to be all ones.
	public void InitBeta(double norm, int index) {
		Arrays.fill(this.beta[index], 1.0/norm); 
	}
	
	
	public void ComputeAllAlphas(double[][] local, double[] theta, double epsilon) 
	{
		for (int i = 1; i < local.length; i++) 
		{
			ComputeSingleAlpha(local[i], theta, epsilon, i, i-1, i);
		}
	}
	
	public void ComputeSingleAlpha(double[] local_t, double[] theta, double epsilon, int norm_index, int t_1, int t) 
	{
		norm_factor[norm_index] = 0.0;
		for (int s = 0; s < this.number_of_topic; s++) {
			this.alpha[t][s] = epsilon*theta[s]*local_t[s];  // regardless of the previous
			// topic - remember that sum_k alpha[t-1][k] is 1 (because of the norm).
			this.alpha[t][s+this.number_of_topic] = (1-epsilon)*(alpha[t_1][s] + alpha[t_1][s+this.number_of_topic])*local_t[s];
			norm_factor[norm_index] += alpha[t][s]+alpha[t][s+this.number_of_topic];
		}
		
		Utils.scaleArray(alpha[t], 1.0/norm_factor[norm_index]);
	}
	
	
	public void ComputeAllBetas(double[][] local, double[] theta, double epsilon)  {
		for (int i = local.length - 2; i >= 0; i--) {
			ComputeSingleBeta(local[i+1], theta, epsilon, norm_factor[i], i+1, i);
		}
	}
	
	
	// This method computes the betas for a single level after beta has been
	// computed for the next level.
	public void ComputeSingleBeta(double[] local_t_1, double[] theta, double epsilon, double norm, int t1, int t) {
	  double trans_sum = 0;
	  
	  for (int i = 0; i < this.number_of_topic; i++) {
	    trans_sum += epsilon*theta[i]*local_t_1[i]*beta[t1][i];
	  }

	  for (int s = 0; s < this.number_of_topic; s++) {
	    // Recall that beta_t1[s] == beta_t1[s+topics_]
	    beta[t][s] = trans_sum + (1-epsilon)*local_t_1[s]*beta[t1][s];
	    beta[t][s] /= norm;
	    beta[t][s+this.number_of_topic] = beta[t][s];
	  }
	  // we've already normalized the betas!
	}
	
	
	public void CombineAllProbs(double[][] sprobs)
    {
		for (int i = 0; i < alpha.length; i++) {
			CombineSingleProb(i,i, sprobs[i]);
		}
    }
	
	
	// This method combines the alpha and beta to get probabilities for a
	// single level.
	public void CombineSingleProb(int alpha_index, int beta_index, double[] sprobs) {
	  double norm = 0;
	  for (int s = 0; s < this.number_of_states; s++) {
	    sprobs[s] = alpha[alpha_index][s]*beta[beta_index][s];
	    norm += sprobs[s];
	  }
	  Utils.scaleArray(sprobs, 1.0/norm);
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
