package markovmodel;
import java.util.Arrays;

import structures._Doc;
import utils.Utils;

public class FastRestrictedHMM_sentiment extends FastRestrictedHMM {

	double m_sigma;//probability of sentiment switch
	double m_transitMatrix[][][];
	
	public FastRestrictedHMM_sentiment(double epsilon,double sigma, int maxSeqSize, int topicSize) {
		super(epsilon, maxSeqSize, topicSize, 3); // 3 is constant
		m_sigma = sigma;
		m_transitMatrix = new double[maxSeqSize][3*this.number_of_topic][3*this.number_of_topic];
		
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
	
	double sumOfAlphas(int i, int t){
		double sum = Double.NEGATIVE_INFINITY;
		int ti = topicMapper(i), si = sentimentMapper(i);
		if (i<this.number_of_topic) {//both changed
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si!=sentimentMapper(j) && ti!=topicMapper(j))
					sum = Utils.logSum(sum, alpha[t][j] + m_transitMatrix[t][j][i]);
			}
		} else if (i<2*this.number_of_topic) {//only topic changed
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti!=topicMapper(j))
					sum = Utils.logSum(sum, alpha[t][j] + m_transitMatrix[t][j][i]);
			}
		} else {//both stay the same
			sum = Utils.logSum(alpha[t][i-2*this.number_of_topic], alpha[t][i-this.number_of_topic]);
			sum = Utils.logSum(sum, alpha[t][i]);
		}
		return sum;
	}
	
	void generateTransitionMatrix(){
		double logEpsilon, logOneMinusEpsilon;
		double logSigma, logOneMinusSigma;
		double epsilon, sigma;
		
		double sum;
		for(int t=1; t<this.length_of_seq;t++){
			epsilon = getEpsilon(t);
			logEpsilon = Math.log(epsilon);
			logOneMinusEpsilon = Math.log(1.0 - epsilon);
			
			sigma = getSigma(t);
			logSigma = Math.log(sigma);
			logOneMinusSigma = Math.log(1.0 - sigma);
			
			sum = Double.NEGATIVE_INFINITY;
			
			for(int i=0; i<3*this.number_of_topic;i++){
				int ti = topicMapper(i), si = sentimentMapper(i);
				sum = Double.NEGATIVE_INFINITY;
				for(int j=0; j<3*this.number_of_topic; j++){
					int tj = topicMapper(j), sj = sentimentMapper(j);
					if(si!=sj && ti!=tj)
						m_transitMatrix[t][i][j] = logSigma + logEpsilon + m_docPtr.m_topics[tj];
					else if(si==sj && ti!=tj)
						m_transitMatrix[t][i][j] = logOneMinusSigma + logEpsilon + m_docPtr.m_topics[tj];
					else
						m_transitMatrix[t][i][j] = logOneMinusSigma + logOneMinusEpsilon;
					sum = Utils.logSum(sum, m_transitMatrix[t][i][j]);
				}
				// normalizing the transition Matrix
				for(int j=0; j<3*this.number_of_topic; j++)
					m_transitMatrix[t][i][j] -= sum;
			}
		}
	}
	
	@Override
	double forwardComputation(double[][] emission, double[] theta) {
		generateTransitionMatrix();
		double logLikelihood = 0, norm = Double.NEGATIVE_INFINITY;//log0
		
		int previousSentenceSenitment = this.m_docPtr.getSentence(0).getSentenceSenitmentLabel();
		int currentSentenceSenitment;
		
		for (int t = 1; t < this.length_of_seq; t++) {
			norm = Double.NEGATIVE_INFINITY;			
			currentSentenceSenitment = this.m_docPtr.getSentence(t).getSentenceSenitmentLabel();
		
			if(currentSentenceSenitment==-1 || previousSentenceSenitment==-1){
				//this means this documnet is not from newEgg
				// theta is represented as all positive topics then all negative topics
				for (int i = 0; i < this.number_of_topic; i++) {
					alpha[t][i] = emission[t][i] + sumOfAlphas(i, t-1);  
					alpha[t][i+this.number_of_topic] = emission[t][i] + sumOfAlphas(i+this.number_of_topic, t-1);  // same sentiment but different topic
					norm = Utils.logSum(norm, Utils.logSum(alpha[t][i], alpha[t][i+this.number_of_topic]));
					
					alpha[t][i+2*this.number_of_topic] = emission[t][i] + sumOfAlphas(i+2*this.number_of_topic, t-1); // same sentiment and same topic
					norm = Utils.logSum(norm, alpha[t][i+2*this.number_of_topic]);
				}
			}
			else{//this means this document has partial annotation
				if(previousSentenceSenitment!=currentSentenceSenitment){
					//this means we have to consider only the first chunck // both sentiment & topic switch
					for (int i = 0; i < this.number_of_topic; i++) {
						alpha[t][i] = emission[t][i] + sumOfAlphas(i, t-1);
						norm = Utils.logSum(norm, alpha[t][i]);
					}
				}
				else{
					//this means we have to consider only the second & third chunck // topic switch or not
					for (int i = 0; i < this.number_of_topic; i++) {
						alpha[t][i+this.number_of_topic] = emission[t][i] + sumOfAlphas(i+this.number_of_topic, t-1);  // same sentiment but different topic
						alpha[t][i+2*this.number_of_topic] = emission[t][i] + sumOfAlphas(i+2*this.number_of_topic, t-1); // same sentiment and same topic
						norm = Utils.logSum(norm, Utils.logSum(alpha[t][i+this.number_of_topic], alpha[t][i+2*this.number_of_topic]));
					}
				}
			}
		
			previousSentenceSenitment = currentSentenceSenitment;
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
		
		double sum = Double.NEGATIVE_INFINITY, probj;
		int ti, si, tj, sj;
		
		int nextSentenceSenitment = this.m_docPtr.getSentence(this.length_of_seq-1).getSentenceSenitmentLabel();
		int currentSentenceSenitment;
	
		for(int t=this.length_of_seq-2; t>=0; t--) {
			currentSentenceSenitment = this.m_docPtr.getSentence(t).getSentenceSenitmentLabel();

			if(currentSentenceSenitment==-1 || nextSentenceSenitment==-1){
				for (int i = 0; i < this.number_of_topic; i++) {
					ti = topicMapper(i);
					si = sentimentMapper(i);
					sum = Double.NEGATIVE_INFINITY;

					for(int j=0; j<this.constant*this.number_of_topic; j++) {
						tj = topicMapper(j);
						sj = sentimentMapper(j);
						probj = emission[t+1][tj] + beta[t+1][j];
						
						if (sj!=si && tj!=ti)
							sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
						else if (sj==si && tj!=ti)
							sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
						else 
							sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
					}
					sum -= norm_factor[t];

					beta[t][i] = sum;
					beta[t][i + this.number_of_topic] = sum ;
					beta[t][i + 2*this.number_of_topic] = sum;
				}
			} else{
				if(currentSentenceSenitment!=nextSentenceSenitment){
					for (int i = 0; i < this.number_of_topic; i++) {
						ti = topicMapper(i);
						si = sentimentMapper(i);
						sum = Double.NEGATIVE_INFINITY;

						for(int j=0; j<this.number_of_topic; j++) {
							tj = topicMapper(j);
							sj = sentimentMapper(j);
							probj = emission[t+1][tj] + beta[t+1][j];
							if (sj!=si && tj!=ti) {
								sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
							} 
						}
						sum -= norm_factor[t];

						beta[t][i] = sum;
						beta[t][i + this.number_of_topic] = sum ;
						beta[t][i + 2*this.number_of_topic] = sum;
					}
				} else{
					for (int i = 0; i < this.number_of_topic; i++) {
						ti = topicMapper(i);
						si = sentimentMapper(i);
						sum = Double.NEGATIVE_INFINITY;

						for(int j=this.number_of_topic; j<this.constant*this.number_of_topic; j++) {
							tj = topicMapper(j);
							sj = sentimentMapper(j);
							probj = emission[t+1][tj] + beta[t+1][j];
							if (sj==si && tj!=ti) {
								sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
							} else {
								sum = Utils.logSum(sum, m_transitMatrix[t][i][j] + probj);
							}
						}
						sum -= norm_factor[t];

						beta[t][i] = sum;
						beta[t][i + this.number_of_topic] = sum ;
						beta[t][i + 2*this.number_of_topic] = sum;
					}
				}
			}
			if(Double.isInfinite(sum))
				System.out.println(sum);
			nextSentenceSenitment = currentSentenceSenitment;
		}
	}
	
	//-----------------Viterbi Algorithm--------------------//
	//NOTE: all computation in log space
	@Override
	public void computeViterbiAlphas(double[][] emission, double[] theta) {
		int prev_best;
		int ti;
		
		for (int t = 1; t < this.length_of_seq; t++) {			
			for (int i = 0; i < this.number_of_topic; i++) {
				prev_best = FindBestInLevel(t-1, i);
				alpha[t][i] = alpha[t-1][prev_best] + m_transitMatrix[t][prev_best][i] + emission[t][i];
				beta[t][i] = prev_best;
				
				prev_best = FindBestInLevel(t-1, i+this.number_of_topic);
				ti = topicMapper(i+this.number_of_topic);
				alpha[t][i+this.number_of_topic] = alpha[t-1][prev_best] +  m_transitMatrix[t][prev_best][i+this.number_of_topic]  + emission[t][i];
				beta[t][i+this.number_of_topic] = prev_best;
				
				prev_best = FindBestInLevel(t-1, i+2*this.number_of_topic);
				alpha[t][i+2*this.number_of_topic] = alpha[t-1][prev_best] + m_transitMatrix[t][prev_best][i+2*this.number_of_topic]  + emission[t][i];
				beta[t][i+2*this.number_of_topic] = prev_best;
				
			}// End for i
		}//End For t
	}
	
	int FindBestInLevel(int t, int i) {
		
		int ti = topicMapper(i), si = sentimentMapper(i);
		double best=Double.NEGATIVE_INFINITY;
		int best_index = -1;
		
		if (i<this.number_of_topic) {//both changed

			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si!=sentimentMapper(j) && ti!=topicMapper(j)){
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		} else if (i<2*this.number_of_topic) {//only topic changed
			
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti!=topicMapper(j)){
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		} else {//both stay the same
			
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti==topicMapper(j)){
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		}		
		return best_index;
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
	
	@Override
	public void BackTrackBestPath(_Doc d, double[][] emission, int[] path) {
		this.length_of_seq = d.getSenetenceSize();
		
		initAlpha(d.m_topics,emission[0]);
		computeViterbiAlphas(emission, d.m_topics);
		
		int level = this.length_of_seq - 1;
		path[level] = FindBestInLevel(level);
		for(int i = this.length_of_seq - 2; i>=0; i--)
			path[i] = (int)beta[i+1][path[i+1]]; 
		
		for(int i=0; i<this.length_of_seq; i++){
			int predictedSentiment = sentimentMapper(topicMapper(path[i]));
			d.getSentence(i).setSentencePredictedSenitmentLabel(predictedSentiment);
		}
	}
}
