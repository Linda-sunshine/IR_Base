package markovmodel;

import java.util.Arrays;

import structures._Doc;
import utils.Utils;

public class FastRestrictedHMM_sentiment extends FastRestrictedHMM{

	double m_sigma;
	
	public FastRestrictedHMM_sentiment(double epsilon,double sigma, int maxSeqSize, int topicSize) {
			
		super(epsilon, maxSeqSize, topicSize);
		m_sigma = sigma;
		alpha  = new double[maxSeqSize][3*this.number_of_topic];
		beta = new double[maxSeqSize][3*this.number_of_topic];
	}
		
	public void setSigma(double sigma){
		m_sigma = sigma;
	}
	//NOTE: in real space!!!!
	double getSigma(int t){
		return m_sigma;
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
			
			for (int i = 0; i < this.number_of_topic/2; i++) {
				
				int index = -1;
				if(i== this.number_of_topic/2 - 1)
					index = i + 1;
				else
					index = i+this.number_of_topic/2 + 1;
					
				alpha[t][i] = logSigma + logEpsilon + theta[index] + emission[t][index];  // change sentiment so change also the topics
				
				if(i== this.number_of_topic/2 - 1)
					index = 0;
				else
					index = i + 1;
				
				
				alpha[t][i+this.number_of_topic] = logOneMinusSigma + logEpsilon + theta[index] + emission[t][index];  // same sentiment but different topic
				
				alpha[t][i+2*this.number_of_topic] = logOneMinusSigma + logOneMinusEpsilon + Utils.logSum(Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]), alpha[t-1][i+2*this.number_of_topic]) + emission[t][i]; // same sentiment and same topic
				
				norm = Utils.logSum(norm, Utils.logSum(Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]), alpha[t][i+2*this.number_of_topic]));
			}
			
			
			for (int i = this.number_of_topic/2; i < this.number_of_topic; i++) {
				
				int index = -1;
				if(i== this.number_of_topic)
					index = this.number_of_topic/2 - 1;
				else
					index = i - this.number_of_topic/2 + 1;
				
	
				alpha[t][i] = logSigma + logEpsilon + theta[index] + emission[t][index];  // change sentiment so change also the topics
				
				

				if(i== this.number_of_topic)
					index = this.number_of_topic/2;
				else
					index = i + 1;
				
				alpha[t][i+this.number_of_topic] = logOneMinusSigma + logEpsilon + theta[index] + emission[t][index];  // same sentiment but different topic
				
				alpha[t][i+2*this.number_of_topic] = logOneMinusSigma + logOneMinusEpsilon + Utils.logSum(Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]), alpha[t-1][i+2*this.number_of_topic]) + emission[t][i]; // same sentiment and same topic
				
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
	
	@Override
	void backwardComputation(double[][] emission, double[] theta) {
		//initiate beta_n
		Arrays.fill(beta[this.length_of_seq-1], 0);
		
		double sum, logEpsilon, logOneMinusEpsilon, logSigma, logOneMinusSigma;
		for(int t=this.length_of_seq-2; t>=0; t--) {
			logEpsilon = Math.log(getEpsilon(t+1));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t+1));
			logSigma = Math.log(getSigma(t+1));
			logOneMinusSigma = Math.log(1.0 - getSigma(t+1));
			
			sum = Double.NEGATIVE_INFINITY;//log0
			for (int j = 0; j < this.number_of_topic; j++)
				sum = Utils.logSum(sum, theta[j] + emission[t+1][j] + beta[t+1][j]);
			sum += (logEpsilon + logSigma);
			
			double sum1;
			sum1 = Double.NEGATIVE_INFINITY;//log0
			for (int j = 0; j < this.number_of_topic; j++)
				sum1 = Utils.logSum(sum1, theta[j] + emission[t+1][j] + beta[t+1][j]);
			sum1 += (logEpsilon + logOneMinusSigma);
			
			
			
			for (int i = 0; i < this.number_of_topic/2; i++) {
				
				int index = -1;
				if(i== this.number_of_topic/2 - 1)
					index = i + 1;
				else
					index = i+this.number_of_topic/2 + 1;
				
				beta[t][i] = Utils.logSum(logOneMinusEpsilon +logOneMinusSigma + beta[t+1][index] + emission[t+1][index], sum) - norm_factor[t];
				
				
				if(i== this.number_of_topic/2 - 1)
					index = 0;
				else
					index = i + 1;
				
				beta[t][i + this.number_of_topic] = Utils.logSum(logOneMinusEpsilon +logSigma+ beta[t+1][index] + emission[t+1][index], sum1) - norm_factor[t];
				beta[t][i + 2*this.number_of_topic] = beta[t][i + this.number_of_topic];
			}
			
			
			
			for (int i = this.number_of_topic/2; i < this.number_of_topic; i++) {
				int index = -1;
				if(i== this.number_of_topic)
					index = this.number_of_topic/2 - 1;
				else
					index = i - this.number_of_topic/2 + 1;
				beta[t][i] = Utils.logSum(logOneMinusEpsilon +logOneMinusSigma + beta[t+1][index] + emission[t+1][index], sum) - norm_factor[t];
				
				
				if(i== this.number_of_topic)
					index = this.number_of_topic/2;
				else
					index = i + 1;
				
				
				
				beta[t][i + this.number_of_topic] = Utils.logSum(logOneMinusEpsilon +logSigma+ beta[t+1][index] + emission[t+1][index], sum1) - norm_factor[t];
				beta[t][i + 2*this.number_of_topic] = beta[t][i + this.number_of_topic];
			}
			
			
		}
	}
	
	@Override
	public void collectExpectations(double[][] sstat) {
		for(int t=0; t<this.length_of_seq; t++) {
			double norm = Double.NEGATIVE_INFINITY;//log0
			for(int i=0; i<3*this.number_of_topic; i++) 
				norm = Utils.logSum(norm, alpha[t][i] + beta[t][i]);
			
			for(int i=0; i<3*this.number_of_topic; i++) 
				sstat[t][i] = Math.exp(alpha[t][i] + beta[t][i] - norm); // convert into original space
		}
	}
	
	//-----------------Viterbi Algorithm--------------------//
	//NOTE: all computation in log space
	@Override
	public void computeViterbiAlphas(double[][] emission, double[] theta) {
		double logEpsilon, logOneMinusEpsilon,logSigma, logOneMinusSigma;
		
		for (int t = 1; t < this.length_of_seq; t++) {
			logEpsilon = Math.log(getEpsilon(t));
			logOneMinusEpsilon = Math.log(1.0 - getEpsilon(t));
			
			logSigma = Math.log(getSigma(t));
			logOneMinusSigma = Math.log(1.0 - getSigma(t));
			
			int prev_best = FindBestInLevel(t-1);
			for (int i = 0; i < this.number_of_topic/2; i++) {
				
				
				int index = -1;
				if(i== this.number_of_topic/2 - 1)
					index = i + 1;
				else
					index = i+this.number_of_topic/2 + 1;
				
				//\psi=1: and \sigma= 1random sample a topic
				alpha[t][i] = alpha[t-1][prev_best] + theta[index] + emission[t][index] + logEpsilon + logSigma;
				beta[t][i] = prev_best;
				
				
				if(i== this.number_of_topic/2 - 1)
					index = 0;
				else
					index = i + 1;
				
				
				////\psi=1: and \sigma= 0 random sample a topic
				alpha[t][i+this.number_of_topic] = alpha[t-1][prev_best] + theta[index] + emission[t][index] + logEpsilon + logOneMinusSigma;
				beta[t][i+this.number_of_topic] = prev_best;
				
					
				//\psi=0 and \sigma = 0 keep previous topic
				if((alpha[t-1][i] > alpha[t-1][i+this.number_of_topic]) && (alpha[t-1][i] > alpha[t-1][i+2*this.number_of_topic])) {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+this.number_of_topic] = i;
				} else if ((alpha[t-1][i+this.number_of_topic] > alpha[t-1][i]) && (alpha[t-1][i+this.number_of_topic] > alpha[t-1][i+2*this.number_of_topic])){
					alpha[t][i+this.number_of_topic] = alpha[t-1][i+this.number_of_topic] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+this.number_of_topic] = i + this.number_of_topic;
				} else if((alpha[t-1][i+2*this.number_of_topic] > alpha[t-1][i]) && (alpha[t-1][i+2*this.number_of_topic] > alpha[t-1][i+this.number_of_topic]))
				{
					alpha[t][i+2*this.number_of_topic] = alpha[t-1][i+2*this.number_of_topic] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+2*this.number_of_topic] = i + 2*this.number_of_topic;
				}
			}// End for i
			
			
		for (int i = this.number_of_topic/2; i < this.number_of_topic; i++) {
				
				
				int index = -1;
				if(i== this.number_of_topic)
					index = this.number_of_topic/2 - 1;
				else
					index = i - this.number_of_topic/2 + 1;
	
				//\psi=1: and \sigma= 1random sample a topic
				alpha[t][i] = alpha[t-1][prev_best] + theta[index] + emission[t][index] + logEpsilon + logSigma;
				beta[t][i] = prev_best;
				
				if(i== this.number_of_topic)
					index = this.number_of_topic/2;
				else
					index = i + 1;
			
				
				////\psi=1: and \sigma= 0 random sample a topic
				alpha[t][i+this.number_of_topic] = alpha[t-1][prev_best] + theta[index] + emission[t][index] + logEpsilon + logOneMinusSigma;
				beta[t][i+this.number_of_topic] = prev_best;
				
					
				//\psi=0 and \sigma = 0 keep previous topic
				if((alpha[t-1][i] > alpha[t-1][i+this.number_of_topic]) && (alpha[t-1][i] > alpha[t-1][i+2*this.number_of_topic])) {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+this.number_of_topic] = i;
				} else if ((alpha[t-1][i+this.number_of_topic] > alpha[t-1][i]) && (alpha[t-1][i+this.number_of_topic] > alpha[t-1][i+2*this.number_of_topic])){
					alpha[t][i+this.number_of_topic] = alpha[t-1][i+this.number_of_topic] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+this.number_of_topic] = i + this.number_of_topic;
				} else if((alpha[t-1][i+2*this.number_of_topic] > alpha[t-1][i]) && (alpha[t-1][i+2*this.number_of_topic] > alpha[t-1][i+this.number_of_topic]))
				{
					alpha[t][i+2*this.number_of_topic] = alpha[t-1][i+2*this.number_of_topic] + logOneMinusEpsilon +logOneMinusSigma + emission[t][i];
					beta[t][i+2*this.number_of_topic] = i + 2*this.number_of_topic;
				}
			}// End for i
			
		}//End For t
	}

	@Override
	int FindBestInLevel(int t) {
		double best = alpha[t][0];
		int best_index = 0;
		for(int i = 1; i<3*this.number_of_topic; i++){
			if(alpha[t][i] > best){
				best = alpha[t][i];
				best_index = i;
			}
		}
		return best_index;
	}

}
