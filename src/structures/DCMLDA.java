package structures;

import topicmodels.LDA_Gibbs_Debug;

public class DCMLDA extends LDA_Gibbs_Debug{
	public DCMLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
		
		
	}
	
	
}
