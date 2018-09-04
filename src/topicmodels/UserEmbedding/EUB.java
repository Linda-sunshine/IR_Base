package topicmodels.UserEmbedding;

import structures._Corpus;
import topicmodels.LDA.LDA_Variational;

/***
 * @Auther Lin Gong
 * The joint modeling of user embedding and topic embedding
 */

public class EUB extends LDA_Variational {

	public EUB(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			int varMaxIter, double varConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
				varMaxIter, varConverge);
	}

	// EMonCorpus will be called in TopicModel
    public void EMonCorpus(){

    }



	
}