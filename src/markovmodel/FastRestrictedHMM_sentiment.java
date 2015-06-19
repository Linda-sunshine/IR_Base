package markovmodel;
import java.util.Arrays;
import utils.Utils;

public class FastRestrictedHMM_sentiment extends FastRestrictedHMM {

	double m_sigma;//probability of sentiment switch
	
	public FastRestrictedHMM_sentiment(double epsilon,double sigma, int maxSeqSize, int topicSize) {
		super(epsilon, maxSeqSize, topicSize, 3); // 3 is constant
		
		m_sigma = sigma;
	}
	
	public void setSigma(double sigma){
		m_sigma = sigma;
	}
	
	//NOTE: in real space!!!!
	double getSigma(int t){
		return m_sigma;
	}	
	
	//one half for the first sentiment, second half for the second sentiment
	public int sentimentMapper(int i) {
		int range = this.number_of_topic / 2;//we assumed the topic size has to been even!!!
		return topicMapper(i) / range;
	}
	
	public int topicMapper(int i) {
		return i%this.number_of_topic;
	}	
	
	double sumOfAlphas(int i, int t, double[] theta){
		double sum = Double.NEGATIVE_INFINITY;
		int ti = topicMapper(i), si = sentimentMapper(i);
		if (i<this.number_of_topic) {//both changed
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si!=sentimentMapper(j) && ti!=topicMapper(j))
					sum = Utils.logSum(sum, alpha[t][j] + theta[ti]);
			}
		} else if (i<2*this.number_of_topic) {//only topic changed
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti!=topicMapper(j))
					sum = Utils.logSum(sum, alpha[t][j] + theta[ti]);
			}
		} else {//both stay the same
			sum = Utils.logSum(alpha[t][i-2*this.number_of_topic], alpha[t][i-this.number_of_topic]);
			sum = Utils.logSum(sum, alpha[t][i]);
		}
		return sum;
	}
	
	@Override
	double forwardComputation(double[][] emission, double[] theta) {
		double epsilon, logEpsilon, logOneMinusEpsilon;
		double sigma, logSigma, logOneMinusSigma;
		double logLikelihood = 0, norm = Double.NEGATIVE_INFINITY;//log0
		
		for (int t = 1; t < this.length_of_seq; t++) {
			epsilon = getEpsilon(t);
			logEpsilon = Math.log(epsilon);
			logOneMinusEpsilon = Math.log(1.0 - epsilon);
			
			sigma = getSigma(t);
			logSigma = Math.log(sigma);
			logOneMinusSigma = Math.log(1.0 - sigma);
			
			// theta is represented as all positive topics then all negative topics
			for (int i = 0; i < this.number_of_topic; i++) {
				alpha[t][i] = logSigma + logEpsilon + emission[t][i] + sumOfAlphas(i, t-1, theta);  
				
				alpha[t][i+this.number_of_topic] = logOneMinusSigma + logEpsilon + emission[t][i] + sumOfAlphas(i+this.number_of_topic, t-1, theta);  // same sentiment but different topic
				norm = Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]);
				
				alpha[t][i+2*this.number_of_topic] = logOneMinusSigma + logOneMinusEpsilon + emission[t][i] + sumOfAlphas(i+2*this.number_of_topic, t-1, theta); // same sentiment and same topic
				norm = Utils.logSum(norm, alpha[t][i+2*this.number_of_topic]);
			}
		
			//normalization
			for (int i = 0; i < this.constant*this.number_of_topic; i++)
				alpha[t][i] -= norm;
			
			logLikelihood += norm; 
			norm_factor[t] = norm;
		}
		return logLikelihood;
	}
	
	@Override
	void backwardComputation(double[][] emission, double[] theta) {
		//initiate beta_n
		Arrays.fill(beta[this.length_of_seq-1], 0);
		
		double epsilon, logEpsilon, logOneMinusEpsilon;
		double sigma, logSigma, logOneMinusSigma;
		double sum = Double.NEGATIVE_INFINITY, probj;
		int ti, si, tj, sj;
		
		for(int t=this.length_of_seq-2; t>=0; t--) {
			epsilon = getEpsilon(t);
			logEpsilon = Math.log(epsilon);
			logOneMinusEpsilon = Math.log(1.0 - epsilon);
			
			sigma = getSigma(t);
			logSigma = Math.log(sigma);
			logOneMinusSigma = Math.log(1.0 - sigma);
			
			for (int i = 0; i < this.number_of_topic; i++) {
				ti = topicMapper(i);
				si = sentimentMapper(i);
				sum = Double.NEGATIVE_INFINITY;
				
				for(int j=0; j<this.constant*this.number_of_topic; j++) {
					tj = topicMapper(j);
					sj = sentimentMapper(j);
					probj = emission[t+1][tj] + beta[t+1][j];
					if (sj!=si && tj!=ti) {
						sum = Utils.logSum(sum, logSigma + logEpsilon + theta[tj] + probj);
					} else if (sj==si && tj!=ti) {
						sum = Utils.logSum(sum, logOneMinusSigma + logEpsilon + theta[tj] + probj);
					} else {
						sum = Utils.logSum(sum, logOneMinusSigma + logOneMinusEpsilon + probj);
					}
				}
				sum -= norm_factor[t];
				
				beta[t][i] = sum;
				beta[t][i + this.number_of_topic] = sum ;
				beta[t][i + 2*this.number_of_topic] = sum;
			}
		}
	}
}
