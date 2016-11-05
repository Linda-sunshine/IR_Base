package topicmodels.LDA;

import structures._Corpus;

public class sparseLDA extends LDA_Gibbs {
	public sparseLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

	}

}
