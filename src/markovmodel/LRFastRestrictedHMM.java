package markovmodel;

import structures._Doc;
import utils.Utils;

public class LRFastRestrictedHMM extends FastRestrictedHMM {
	
	double[] m_omega; // feature weight for topic transition
	double[] m_epsilons; // topic transition for each sentence
	
	public LRFastRestrictedHMM (double[] omega) {
		super();
		m_omega = omega;
	}

	public double ForwardBackward(_Doc d, double[][] emission) {
		this.number_of_topic = d.m_topics.length;
		this.length_of_seq = d.getSenetenceSize();
		
		alpha  = new double[this.length_of_seq][2*this.number_of_topic];
		beta = new double[this.length_of_seq][2*this.number_of_topic];
		norm_factor = new double[this.length_of_seq];
		initEpsilons(d);		
		
		double loglik = initAlpha(d.m_topics, emission[0]) + forwardComputation(emission, d.m_topics);
		backwardComputation(emission, d.m_topics);		
		
		return loglik;
	}
	
	//all epsilons in real space!!
	void initEpsilons(_Doc d) {
		m_epsilons = new double[this.length_of_seq];
		for(int t=1; t<this.length_of_seq; t++)
			m_epsilons[t] = Utils.logistic(d.m_sentence_features[t-1], m_omega, 1);//first sentence does not have features
	}
	
	//epsilon in log-space
	double forwardComputation(double[][] emission, double[] theta) {
		double logLikelihood = 0, norm, logEpsilon;
		for (int t = 1; t < this.length_of_seq; t++) {
			norm = Double.NEGATIVE_INFINITY;//log0
			logEpsilon = Math.log(m_epsilons[t]);
			logOneMinusEpsilon = Math.log(1.0-m_epsilons[t]);
			
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = logEpsilon + theta[i] + emission[t][i];  // regardless of the previous
				alpha[t][i+this.number_of_topic] = logOneMinusEpsilon + Utils.logSum(alpha[t-1][i], alpha[t-1][i+this.number_of_topic]) + emission[t][i];
				norm = Utils.logSum(norm, Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]));
			}
			
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] -= norm;
				alpha[t][i+this.number_of_topic] -= norm;
			}
			
			logLikelihood += norm; 
			norm_factor[t] = norm;
		}
		return logLikelihood;
	}
	
	void backwardComputation(double[][] emission, double[] theta) {
		double logEpsilon;
		for(int t=this.length_of_seq-2; t>=0; t--) {
			logEpsilon = Math.log(m_epsilons[t+1]); // should use next sentence's epsilon
			logOneMinusEpsilon = Math.log(1.0 - m_epsilons[t+1]);
			
			double sum = Double.NEGATIVE_INFINITY;//log0
			for (int j = 0; j < this.number_of_topic; j++)
				sum = Utils.logSum(sum, theta[j] + emission[t+1][j] + beta[t+1][j]);
			sum += logEpsilon;
			
			for (int i = 0; i < this.number_of_topic; i++) {
				beta[t][i] = Utils.logSum(logOneMinusEpsilon + beta[t+1][i] + emission[t+1][i], sum) - norm_factor[t];
				beta[t][i + this.number_of_topic] = beta[t][i];
			}
		}
	}
	
	//-----------------Viterbi Algorithm--------------------//
	public void ComputeAllalphas(double[][] emission, double[] theta) {
		double norm, logEpsilon;
		for (int t = 1; t < this.length_of_seq; t++) {
			int prev_best = FindBestInLevel(t-1);
			norm = Double.NEGATIVE_INFINITY;//log0
			logEpsilon = Math.log(m_epsilons[t]);
			logOneMinusEpsilon = Math.log(1.0 - m_epsilons[t]);
			
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = alpha[t-1][prev_best] + theta[i] + emission[t][i] + logEpsilon;
				best[t][i] = prev_best;
				
				if(alpha[t-1][i] > alpha[t-1][i+this.number_of_topic]) {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i] + logOneMinusEpsilon + emission[t][i];
					best[t][i+this.number_of_topic] = i;
				} else {
					alpha[t][i+this.number_of_topic] = alpha[t-1][i+this.number_of_topic] + logOneMinusEpsilon + emission[t][i];
					best[t][i+this.number_of_topic] = i+this.number_of_topic;
				}
				norm = Utils.logSum(norm, Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]));
			}// End for i

			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] -= norm;
				alpha[t][i+this.number_of_topic] -= norm;
			}
		}//End For t
	}
	
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		this.number_of_topic = d.m_topics.length;
		this.length_of_seq = d.getSenetenceSize();
		alpha  = new double[this.length_of_seq][2*this.number_of_topic];
		this.best = new int[this.length_of_seq][2*this.number_of_topic];
		
		initEpsilons(d);
		initAlpha(d.m_topics,emission[0]);
		ComputeAllalphas(emission, d.m_topics);
		
		int level = this.length_of_seq - 1;
		path[level] = FindBestInLevel(level);
		for(int i = this.length_of_seq - 2; i>=0; i--)
			path[i] = best[i+1][path[i+1]];
	}
	
}
