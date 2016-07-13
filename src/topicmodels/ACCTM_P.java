package topicmodels;

import java.util.Collection;

import structures._Corpus;
import structures._Doc;

public class ACCTM_P extends ACCTM_C {
	public ACCTM_P(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma, ksi, tau);

	}
	
	public String toString() {
		return String.format("ACCTM_P topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);

	}
}
