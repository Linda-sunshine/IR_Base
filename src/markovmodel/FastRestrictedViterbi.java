package markovmodel;

import structures._Doc;
import utils.Utils;

public class FastRestrictedViterbi {

	int number_of_topic;
	int length_of_seq;
	double delta [][];
	int best[][]; 
	double logOneMinusEpsilon;
	
	public FastRestrictedViterbi(_Doc d, double epsilon, double[][] emission)
	{
		this.number_of_topic = d.m_topics.length;
		// T = this.length_of_seq is number of sentence in HTMM as number of observation in HMM
		this.length_of_seq = d.getSenetenceSize(); 
		this.delta = new double [this.length_of_seq][2*this.number_of_topic];
		this.best = new int [this.length_of_seq][2*this.number_of_topic];
		logOneMinusEpsilon = Math.log(1.0 - Math.exp(epsilon));
		InitDelta(d.m_topics,emission[0]);
		ComputeAllDeltas(emission, d.m_topics, epsilon);
	}
	
	//NOTE: all computation in log space
	// theta is actually init matrix of HMM
	// local0 is emission[0]
	public void InitDelta(double[] theta, double[] local0)
	{
		double norm = Double.NEGATIVE_INFINITY;//log0
		for (int i = 0; i < this.number_of_topic; i++) {
			this.delta[0][i] = theta[i] + local0[i];
			this.delta[0][i+this.number_of_topic] = Double.NEGATIVE_INFINITY;;
			norm = Utils.logSum(norm, delta[0][i]);
		}
		
		//normalization
		for (int i = 0; i < this.number_of_topic; i++) {
			delta[0][i] -= norm;
			delta[0][i+this.number_of_topic] -= norm;
		}
	}
	
	public void ComputeAllDeltas(double[][] emission, double[] theta, double epsilon)
	{
		for (int t = 1; t < this.length_of_seq; t++) {
			int prev_best = FindBestInLevel(t-1);
			double norm = Double.NEGATIVE_INFINITY;//log0
			for (int i = 0; i < this.number_of_topic; i++) {
				delta[t][i] = delta[t-1][prev_best] + theta[i] + emission[t][i] + epsilon;
				best[t][i] = prev_best;
				if(delta[t-1][i] > delta[t-1][i+this.number_of_topic]){
					delta[t][i+this.number_of_topic] = delta[t-1][i] + logOneMinusEpsilon + emission[t][i];
					best[t][i+this.number_of_topic] = i;
				}
				else{
					delta[t][i+this.number_of_topic] = delta[t-1][i+this.number_of_topic] + logOneMinusEpsilon + emission[t][i];
					best[t][i+this.number_of_topic] = i+this.number_of_topic;
				}
				norm = Utils.logSum(norm, Utils.logSum(delta[t][i], delta[t][i+this.number_of_topic]));
			}// End for i
			
			//normalization
			for (int i = 0; i < this.number_of_topic; i++) {
				delta[t][i] -= norm;
				delta[t][i+this.number_of_topic] -= norm;
			}
		}//End For t
	}
		
	private int FindBestInLevel(int t)
	{
		double best = Double.NEGATIVE_INFINITY;
		int best_index = -1;
		for(int i = 0; i<2*this.number_of_topic; i++){
			if(delta[t][i] > best){
				best = delta[t][i];
				best_index = i;
			}
		}
		return best_index;
	}
	
	
	public void BackTrackBestPath(int[] path)
	{
		// level = this.length_of_seq - 1 (T - 1)
		int level = this.length_of_seq - 1;
		path[level] = FindBestInLevel(level);
		for(int i = this.length_of_seq - 2; i>=0; i--){
			path[i] = best[i+1][path[i+1]];  
		}
	}
	
}
