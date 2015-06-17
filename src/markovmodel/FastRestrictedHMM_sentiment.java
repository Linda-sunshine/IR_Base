package markovmodel;
import java.util.Arrays;
import utils.Utils;

public class FastRestrictedHMM_sentiment extends FastRestrictedHMM{

	double m_sigma;
	
	public FastRestrictedHMM_sentiment(double epsilon,double sigma, int maxSeqSize, int topicSize) {
		super(epsilon, maxSeqSize, topicSize, 3); // 3 is constant
		m_sigma = sigma;
		alpha  = new double[maxSeqSize][this.constant*this.number_of_topic];
		beta = new double[maxSeqSize][this.constant*this.number_of_topic];
	}
	
	public void setSigma(double sigma){
		m_sigma = sigma;
	}
	//NOTE: in real space!!!!
	double getSigma(int t){
		return m_sigma;
	}
	
	
	// return true if sentiment of ith topic is same as sentiment of jth topic
	public boolean sentiment_mapper(int i, int j)
	{
		int topic_i = i%this.number_of_topic;
		int topic_j = j%this.number_of_topic;
		int range = this.number_of_topic / 2;
		return (topic_i/range) == (topic_j/range);
	}
	
	//return true if topic of i is same topic of j
	public boolean topic_mapper(int i, int j)
	{
		int topic_i = i%this.number_of_topic;
		int topic_j = j%this.number_of_topic;
		return topic_i == topic_j;
	}
	
	
	//NOTE: all computation in log space
	@Override
	double initAlpha(double[] theta, double[] local0) {
		double norm = Double.NEGATIVE_INFINITY;//log0
		for (int i = 0; i < this.number_of_topic; i++) {
			alpha[0][i] = local0[i] + theta[i];
			alpha[0][i+this.number_of_topic] = Double.NEGATIVE_INFINITY;//document must start with a new topic
			alpha[0][i+2*this.number_of_topic] = Double.NEGATIVE_INFINITY;
			//this is full computation, but no need to do so
			//norm = Utils.logSum(norm, Utils.logSum(alpha[0][i], alpha[0][i+this.number_of_topic]));
			norm = Utils.logSum(norm, alpha[0][i]);
		}
		
		//normalization
		for (int i = 0; i < this.number_of_topic; i++) {
			alpha[0][i] -= norm;
			//this.alpha[0][i+this.number_of_topic] -= norm; // no need to compute this
		}
		
		norm_factor[0] = norm;
		return norm;
	}
	
	// input theta and alpha in logscale 
	// return data in logscale
	// t is the step number used in alpha
	// level = 1 means sentiment and topic both switch
	// level = 2 means only topic switch
	double sum_alpha(int i, int t, double[] theta, int level){
		
		double sum = 0;
		int topic = i%this.number_of_topic; // calculate the actual topic number
		
		for(int j=0; j<constant*this.number_of_topic; j++)
		{
			if(level==1 && topic_mapper(i, j)==false && sentiment_mapper(i, j)==false)
				sum+= Math.exp(alpha[t][j]) * Math.exp(theta[topic]);
			if(level==2 && topic_mapper(i, j)==false && sentiment_mapper(i, j)==true)
				sum+= Math.exp(alpha[t][j]) * Math.exp(theta[topic]);
		}
		
		return Math.log(sum);
	}
	
	@Override
	double forwardComputation(double[][] emission, double[] theta) {
		double logLikelihood = 0, norm, logEpsilon, logOneMinusEpsilon, logSigma, logOneMinusSigma;
		for (int t = 1; t < this.length_of_seq; t++) {
			norm = Double.NEGATIVE_INFINITY;//log0
			logEpsilon = Math.log(getEpsilon(t));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t));
			logSigma = Math.log(getSigma(t));
			logOneMinusSigma = Math.log(1.0 - getSigma(t));
			// theta is represented as all positive topics then all negative topics
			
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = logSigma + logEpsilon + emission[t][i] + sum_alpha(i, t - 1, theta, 1);  
				alpha[t][i+this.number_of_topic] = logOneMinusSigma + logEpsilon + emission[t][i] + sum_alpha(i+this.number_of_topic, t-1, theta, 2);  // same sentiment but different topic
				alpha[t][i+2*this.number_of_topic] = logOneMinusSigma + logOneMinusEpsilon + emission[t][i] + Utils.logSum(Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]), alpha[t-1][i+2*this.number_of_topic]); // same sentiment and same topic
				norm = Utils.logSum(norm, Utils.logSum(Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]), alpha[t][i+2*this.number_of_topic]));
			}
		
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] -= norm;
				alpha[t][i+this.number_of_topic] -= norm;
				alpha[t][i+2*this.number_of_topic] -= norm;
			}
			
			logLikelihood += norm; 
			norm_factor[t] = norm;
		}
		return logLikelihood;
	}
	
	double sum_beta(int i, int t, double[] theta, double[][] emission, int level){
		
		double sum = 0;
		if(level==1){
			for(int j=0; j<this.number_of_topic-1; j++)
			{
				int topic = j%this.number_of_topic; // calculate the actual topic number
				if(sentiment_mapper(i, j)==false && topic_mapper(i, j)==false)
					sum+= Math.exp(theta[topic])*Math.exp(emission[t][topic])*Math.exp(beta[t][j]);
			}
			sum = sum*m_epsilon*m_sigma;
		}
		
		if(level==2)
		{
			for(int j=this.number_of_topic; j<(constant-1)*this.number_of_topic-1; j++)
			{
				int topic = j%this.number_of_topic; // calculate the actual topic number
				if(sentiment_mapper(i, j)==true && topic_mapper(i, j)==false)
					sum+= Math.exp(theta[topic])*Math.exp(emission[t][topic])*Math.exp(beta[t][j]);
			}
			sum = sum*m_epsilon*(1-m_sigma);
		}
		return Math.log(sum);
	}
	
	
	
	@Override
	void backwardComputation(double[][] emission, double[] theta) {
		//initiate beta_n
		Arrays.fill(beta[this.length_of_seq-1], 0);
		
		double logOneMinusEpsilon, logOneMinusSigma;
		for(int t=this.length_of_seq-2; t>=0; t--) {
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t+1));
			logOneMinusSigma = Math.log(1.0 - getSigma(t+1));
			
			for (int i = 0; i < this.number_of_topic; i++) {
				beta[t][i] = Utils.logSum(Utils.logSum(logOneMinusEpsilon +logOneMinusSigma 
						     + beta[t+1][i+2*this.number_of_topic] + emission[t+1][i], 
						     sum_beta(i, t+1, theta, emission, 1)), sum_beta(i, t+1, theta, emission, 2)) 
						     - norm_factor[t];
				beta[t][i + this.number_of_topic] =  beta[t][(i+this.number_of_topic)%this.number_of_topic] ;
				beta[t][i + 2*this.number_of_topic] = beta[t][(i+2*this.number_of_topic)%this.number_of_topic];
			}
	
		}
	}
	
	@Override
	public void collectExpectations(double[][] sstat) {
		for(int t=0; t<this.length_of_seq; t++) {
			double norm = Double.NEGATIVE_INFINITY;//log0
			for(int i=0; i<this.constant*this.number_of_topic; i++) 
				norm = Utils.logSum(norm, alpha[t][i] + beta[t][i]);
			
			for(int i=0; i<this.constant*this.number_of_topic; i++) 
				sstat[t][i] = Math.exp(alpha[t][i] + beta[t][i] - norm); // convert into original space
		}
	}
	
	@Override
	int FindBestInLevel(int t) {
		double best = alpha[t][0];
		int best_index = 0;
		for(int i = 1; i<this.constant*this.number_of_topic; i++){
			if(alpha[t][i] > best){
				best = alpha[t][i];
				best_index = i;
			}
		}
		return best_index;
	}

}
