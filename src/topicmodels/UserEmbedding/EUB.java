package topicmodels.UserEmbedding;

import structures._Corpus;
import topicmodels.LDA.LDA_Variational;

/***
 *
 */

public class EUB extends LDA_Variational {

	public EUB(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
				varMaxIter, varConverge);
	}


	
}