package topicmodels;

import java.util.Collection;

import structures._Corpus;
import structures._Doc;

public class DCMCorrLDA extends DCMLDA {
	public DCMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, double alpha,
			double burnIn, int lag, int newtonIter, double newtonConverge) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, newtonIter, newtonConverge);
		m_multithread = true;
	}

	public String toString() {
		return String.format("multithread DCMCorrLDA[k:%d]", number_of_topics);
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);

		int cores = Runtime.getRuntime().availableProcessors();
	}
}
