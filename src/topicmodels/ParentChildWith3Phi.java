package topicmodels;

import structures._Corpus;

public class ParentChildWith3Phi extends ParentChild_Gibbs{

	public ParentChildWith3Phi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
		// TODO Auto-generated constructor stub
	}
	
	public void sampleParent()

}
