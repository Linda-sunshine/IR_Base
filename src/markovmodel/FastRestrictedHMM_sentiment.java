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
		return (i%this.number_of_topic) / range;
	}
	
	//topicMapper gives us large i value to the actual topic number
	// for example if total_numner of toics is 6, then we have 3*6=18
	// topics so if the input i is 15, topic Mapper return it as a range between 
	// 0 to 6 which is 15%6 = 3
	public int topicMapper(int i) {
		return (i%this.number_of_topic);
	}
	
	//topicMapper gives us large i value to the actual topic number
	// for example if total_numner of topics is 6 [0 to 2 is positive; 3 to 5 is negative], then we have 3*6=18
	// topics so if the input i is 15, topicIndexMapper return it as a range between 
	// 0 to 3 (half of topic_number) which is (15%6)%(6/2) = 0 here
	// we need because for example P0 cannot transit to P0 or N0
		
	public int topicIndexMapper(int i) {
		int range = this.number_of_topic / 2;
		return (i%this.number_of_topic)%range;
	}
	
	@Override
	//NOTE: all computation in log space
	double initAlpha(double[] theta, double[] local0) {
		
		//initialize all to Double.Negative
		double norm = Double.NEGATIVE_INFINITY;//log0
		for(int t=0; t<this.length_of_seq;t++){
			for(int i=0; i<this.constant*this.number_of_topic; i++){
				alpha[t][i] = Double.NEGATIVE_INFINITY;
			}
		}
		
		//initialize the first coloumn with only the first chunk
		for (int i = 0; i < this.number_of_topic; i++) {
			alpha[0][i] = local0[i] + theta[i];//document must start with a new topic and a new sentiment
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

	double sumOfAlphas(int i, int t){
		double sum = Double.NEGATIVE_INFINITY;
		int ti = topicIndexMapper(i), si = sentimentMapper(i);
		int c=0;
		if (i<this.number_of_topic) {//both changed
			//System.out.print("\nFrom "+t+" to "+(t+1)+" i:"+i+"j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si!=sentimentMapper(j) && ti!=topicIndexMapper(j)){
					//System.out.print(j+",");
					sum = Utils.logSum(sum, alpha[t][j] + m_transitMatrix[t+1][j][i]);
					c++;
				}
			}
		} else if (i<2*this.number_of_topic) {//only topic changed
			//System.out.print("\nFrom "+t+" to "+(t+1)+" i:"+i+"j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti!=topicIndexMapper(j)){
					//System.out.print(j+",");
					sum = Utils.logSum(sum, alpha[t][j] + m_transitMatrix[t+1][j][i]);
					c++;
				}
			}
		} else {//both stay the same
			//System.out.print("\nFrom "+t+" to "+(t+1)+" i:"+i+"j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti==topicIndexMapper(j)){
					//System.out.print(j+",");
					sum = Utils.logSum(sum, alpha[t][j] + m_transitMatrix[t+1][j][i]);
					c++;
				}
			}
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
			
			for(int i=0; i<this.constant*this.number_of_topic;i++){
				int ti = topicIndexMapper(i), si = sentimentMapper(i);
				int tj, sj;
				int docTopicIndex;
				sum = Double.NEGATIVE_INFINITY;
				int c = 0;
				//System.out.print("\ni:"+i+",j: ");
				for(int j=0; j<this.constant*this.number_of_topic; j++){
					tj = topicIndexMapper(j);
					sj = sentimentMapper(j);
					docTopicIndex = topicMapper(j);
					//System.out.println(si+","+sj+","+ti+","+tj+",J:"+j);
					if(j>=0 && j<this.number_of_topic){
						if(si!=sj && ti!=tj)
						{
							//System.out.print(j+",");
							m_transitMatrix[t][i][j] = logSigma + logEpsilon + m_docPtr.m_topics[docTopicIndex];
							c++;
						}
					}
					else if(j>=this.number_of_topic && j<2*this.number_of_topic)
					{
					 if(si==sj && ti!=tj)
						{
						 	//System.out.print(j+",");
						 	m_transitMatrix[t][i][j] = logOneMinusSigma + logEpsilon + m_docPtr.m_topics[docTopicIndex];
							c++;
						}
					}
					else if ((j>=2*this.number_of_topic && j<3*this.number_of_topic)){
						if(si==sj && ti==tj){
							//System.out.print(j+",");
							m_transitMatrix[t][i][j] = logOneMinusSigma + logOneMinusEpsilon;
							c++;
						}
					}
					sum = Utils.logSum(sum, m_transitMatrix[t][i][j]);
					
				}
				//System.out.println("when i is:" + i + ",and c is:" + c);
				// normalizing the transition Matrix
				for(int j=0; j<this.constant*this.number_of_topic; j++)
					m_transitMatrix[t][i][j] -= sum;
			}
		}
	}
	
	@Override
	double forwardComputation(double[][] emission, double[] theta) {
		generateTransitionMatrix();
		double logLikelihood = 0, norm = Double.NEGATIVE_INFINITY;//log0
		
		//System.out.println("Doc Number:"+this.m_docPtr.getID());
		int previousSentenceSenitment = this.m_docPtr.getSentence(0).getSentenceSenitmentLabel();
		int currentSentenceSenitment;
		
		for (int t = 1; t < this.length_of_seq; t++) {
			norm = Double.NEGATIVE_INFINITY;			
			currentSentenceSenitment = this.m_docPtr.getSentence(t).getSentenceSenitmentLabel();
		
			if(currentSentenceSenitment==-1 || previousSentenceSenitment==-1){
				//this means this document is not from newEgg
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
					ti = topicIndexMapper(i);
					si = sentimentMapper(i);
					sum = Double.NEGATIVE_INFINITY;
					//System.out.print("\ni:"+i+",j:");
					for(int j=0; j<this.constant*this.number_of_topic; j++) {
						tj = topicIndexMapper(j);
						sj = sentimentMapper(j);
						int topicj = topicMapper(j);
						probj = emission[t+1][topicj] + beta[t+1][j];
						
						if(j>=0 && j<this.number_of_topic){
							if (sj!=si && tj!=ti){
								sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
								//System.out.print(j+",");
							}
						}
						else if(j>=this.number_of_topic && j<2*this.number_of_topic){
							if (sj==si && tj!=ti){
								sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
								//System.out.print(j+",");
							}
						}
						else if (j>=2*this.number_of_topic && j<3*this.number_of_topic){
							if(sj==si && tj==ti){
								sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
								//System.out.print(j+",");
							}
						}
					}
					sum -= norm_factor[t];

					beta[t][i] = sum;
					beta[t][i + this.number_of_topic] = sum ;
					beta[t][i + 2*this.number_of_topic] = sum;
				}
			} else{
				if(currentSentenceSenitment!=nextSentenceSenitment){
					
					for (int i = 0; i < this.number_of_topic; i++) {
						ti = topicIndexMapper(i);
						si = sentimentMapper(i);
						sum = Double.NEGATIVE_INFINITY;
						int c=0;
						for(int j=0; j<this.number_of_topic; j++) {
							tj = topicIndexMapper(j);
							sj = sentimentMapper(j);
							int topicj = topicMapper(j);
							probj = emission[t+1][topicj] + beta[t+1][j];
							if (sj!=si && tj!=ti) {
								sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
								//System.out.println("Here 1");
								//System.out.println("j is:"+j);
								c++;
							} 
						}
						//System.out.println("c is:"+c+"when i is:"+i);
						sum -= norm_factor[t];

						beta[t][i] = sum;
						beta[t][i + this.number_of_topic] = sum ;
						beta[t][i + 2*this.number_of_topic] = sum;
					}
				} else{
					
					for (int i = 0; i < this.number_of_topic; i++) {
						ti = topicIndexMapper(i);
						si = sentimentMapper(i);
						sum = Double.NEGATIVE_INFINITY;
						int c=0;
						for(int j=this.number_of_topic; j<this.constant*this.number_of_topic; j++) {
							
							tj = topicIndexMapper(j);
							sj = sentimentMapper(j);
							int topicj = topicMapper(j);
							probj = emission[t+1][topicj] + beta[t+1][j];
							if(j>=this.number_of_topic && j<2*this.number_of_topic){
								if (sj==si && tj!=ti) {
									sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
									//System.out.println("Here 2");
									//System.out.println("j is:"+j);
									c++;
								}
							}else if  (j>=2*this.number_of_topic && j<this.constant*this.number_of_topic)
							{	
								if(si==sj && ti==tj){
									sum = Utils.logSum(sum, m_transitMatrix[t+1][i][j] + probj);
									//System.out.println("Here 3");
									//System.out.println("j is:"+j);
									c++;
								}
							}
						}
						//System.out.println("c is:"+c+"when i is:"+i);
						sum -= norm_factor[t];

						beta[t][i] = sum;
						beta[t][i + this.number_of_topic] = sum ;
						beta[t][i + 2*this.number_of_topic] = sum;
					}
				}
			}
			nextSentenceSenitment = currentSentenceSenitment;
		}
	}
	
	//-----------------Viterbi Algorithm--------------------//
	//NOTE: all computation in log space
	@Override
	public void computeViterbiAlphas(double[][] emission, double[] theta) {
		int prev_best;
		for (int t = 1; t < this.length_of_seq; t++) {			
			for (int i = 0; i < this.number_of_topic; i++) {
				prev_best = FindBestInLevel(t-1, i);
				alpha[t][i] = alpha[t-1][prev_best] + m_transitMatrix[t][prev_best][i] + emission[t][i];
				beta[t][i] = prev_best;
				
				prev_best = FindBestInLevel(t-1, i+this.number_of_topic);
				alpha[t][i+this.number_of_topic] = alpha[t-1][prev_best] +  m_transitMatrix[t][prev_best][i+this.number_of_topic]  + emission[t][i];
				beta[t][i+this.number_of_topic] = prev_best;
				
				prev_best = FindBestInLevel(t-1, i+2*this.number_of_topic);
				alpha[t][i+2*this.number_of_topic] = alpha[t-1][prev_best] + m_transitMatrix[t][prev_best][i+2*this.number_of_topic]  + emission[t][i];
				beta[t][i+2*this.number_of_topic] = prev_best;
				
			}// End for i
		}//End For t
	}
	
	int FindBestInLevel(int t, int i) {
		
		int ti = topicIndexMapper(i), si = sentimentMapper(i);
		double best=Double.NEGATIVE_INFINITY;
		int best_index = -1;
		if (i<this.number_of_topic) {//both changed
			//System.out.print("\ni:"+i+", j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si!=sentimentMapper(j) && ti!=topicIndexMapper(j)){
					//System.out.print(j+",");
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		} else if (i<2*this.number_of_topic) {//only topic changed
			//System.out.print("\ni:"+i+", j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti!=topicIndexMapper(j)){
					//System.out.print(j+",");
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		} else {//both stay the same
			//System.out.print("\ni:"+i+", j:");
			for(int j=0; j<this.constant*this.number_of_topic; j++) {
				if(si==sentimentMapper(j) && ti==topicIndexMapper(j)){
					//System.out.print(j+",");
					if(alpha[t][j] > best){
						best = alpha[t][j];
						best_index = j;
					}
				}
			}
		}
		//System.out.println("best:"+best_index);
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
		generateTransitionMatrix();
		computeViterbiAlphas(emission, d.m_topics);
		
		int level = this.length_of_seq - 1;
		path[level] = FindBestInLevel(level);
		for(int i = this.length_of_seq - 2; i>=0; i--)
			path[i] = (int)beta[i+1][path[i+1]]; 
		
		for(int i=0; i<this.length_of_seq; i++){
			int predictedSentiment = sentimentMapper(path[i]);
			d.getSentence(i).setSentencePredictedSenitmentLabel(predictedSentiment);
		}
	}
}
