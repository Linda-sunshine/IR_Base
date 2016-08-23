package topicmodels.correspondenceModels;

import structures._Corpus;

public class DCMDMMCorrLDA_multi_E extends DCMDMMCorrLDA{
	public DCMDMMCorrLDA_multi_E(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha_a,
			double alpha_c, double burnIn, double ksi, double tau, int lag,
			int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, alpha_c, burnIn, ksi, tau, lag, newtonIter, newtonConverge);
	}
	
	
	
}
