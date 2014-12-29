package markovmodel;

import structures._Doc;
import utils.Utils;

public class FastRestrictedHMM {

	int number_of_topic;
	int length_of_seq;
	double alpha[][];
	double beta[][];
	double logOneMinusEpsilon;//to compute log(1-epsilon) efficiently
	
	public FastRestrictedHMM() {
		number_of_topic = 0;
	}
	
	public double ForwardBackward(_Doc d, double epsilon, double [][] local, double [] pi)
	{
		double loglik;
		double[] theta = d.m_topics;
		this.number_of_topic = theta.length;
		this.length_of_seq = d.getSenetenceSize();
		
		alpha  = new double [this.length_of_seq][2*this.number_of_topic];
		beta = new double [this.length_of_seq][2*this.number_of_topic];
		logOneMinusEpsilon = Math.log(1.0 - Math.exp(epsilon));
		
		loglik = initAlpha(pi, local[0]) + forwardComputation(local, theta, epsilon);
		backwardComputation(local, theta, epsilon);		
		
		return loglik;
	}
	
	//NOTE: all computation in log space
	double initAlpha(double[] pi, double[] local0) 
	{
		double norm = Double.NEGATIVE_INFINITY;//log0
		for (int i = 0; i < this.number_of_topic; i++) 
		{
			this.alpha[0][i] = local0[i] + pi[i];
			this.alpha[0][i+this.number_of_topic] = Double.NEGATIVE_INFINITY;
			//this is full computation, but no need to do so
			//norm = Utils.logSum(norm, Utils.logSum(this.alpha[0][i], this.alpha[0][i+this.number_of_topic]));
			norm = Utils.logSum(norm, this.alpha[0][i]);
		}
		
		//normalization
		for (int i = 0; i < this.number_of_topic; i++) {
			this.alpha[0][i] -= norm;
			//this.alpha[0][i+this.number_of_topic] -= norm; // no need to compute this
		}
		
		return norm;
	}
	
	double forwardComputation(double[][] local, double[] theta, double epsilon) 
	{
		double logLikelihood = 0;
		for (int t = 1; t < this.length_of_seq; t++) 
		{
			double norm = Double.NEGATIVE_INFINITY;//log0
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = epsilon + theta[i] + local[t][i];  // regardless of the previous
				this.alpha[t][i+this.number_of_topic] = logOneMinusEpsilon + Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]) + local[t][i];
				
				norm = Utils.logSum(norm, Utils.logSum(this.alpha[t][i], this.alpha[t][i+this.number_of_topic]));
			}
			
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				this.alpha[t][i] -= norm;
				this.alpha[t][i+this.number_of_topic] -= norm;
			}
			
			logLikelihood += norm; 
		}
		return logLikelihood;
	}
	
	void backwardComputation(double[][] local, double[] theta, double epsilon)  {
		for(int t=this.length_of_seq-1; t>=0; t--) {
			double norm = Double.NEGATIVE_INFINITY;//log0
			for (int i = 0; i < this.number_of_topic; i++) {
				double sum = epsilon;
				for (int j = 0; j < this.number_of_topic; j++)
					sum = Utils.logSum(sum, theta[j] + local[t+1][j] + beta[t+1][j]);
				
				beta[t][i] = Utils.logSum(logOneMinusEpsilon + beta[t+1][i] + local[t+1][i], sum);
				beta[t + this.number_of_topic][i] = beta[t][i];
				
				norm = Utils.logSum(norm, Utils.logSum(beta[t][i], beta[t + this.number_of_topic][i]));
			}
			
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				this.beta[t][i] -= norm;
				this.beta[t][i+this.number_of_topic] -= norm;
			}
		}
	}
	
	public void collectExpectations(double[][] sstat) {
		for(int t=0; t<this.length_of_seq; t++) {
			double norm = Double.NEGATIVE_INFINITY;//log0
			for(int i=0; i<2*this.number_of_topic; i++) 
				norm = Utils.logSum(norm, alpha[t][i] + beta[t][i]);
			
			for(int i=0; i<2*this.number_of_topic; i++) 
				sstat[t][i] = Math.exp(alpha[t][i] + beta[t][i] - norm); // convert into original space
		}
	}
}
