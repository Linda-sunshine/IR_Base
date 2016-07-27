package topicmodels;

import structures._Corpus;
import structures._Doc;
import structures._Word;


public class LDA_test extends LDA_Gibbs {
	public LDA_test(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, int number_of_topics,
			double alpha, double burnIn, int lag) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

	}

	
	public double inference(_Doc d) {

		initTest(d);
		
		double logLikelihood = 0;
		logLikelihood = inferenceDoc(d);

		return logLikelihood;

	}
	
	protected void initTest(_Doc d) {

		int testLength = (int) (m_testWord4PerplexityProportion * d
				.getTotalDocLength());
		d.setTopics4GibbsTest(number_of_topics, d_alpha, testLength);
	}
	
	protected double inferenceDoc(_Doc d) {
		double likelihood = 0;
		
		int i=0; 
		do{
			calculate_E_step(d);

			if(i<m_burnIn && i%m_lag==0)
				collectStats(d);
		} while (++i < number_of_iteration);

		estThetaInDoc(d);

		likelihood = calculate_test_log_likelihood(d);

		return likelihood;
	}
	
	protected double calculate_test_log_likelihood(_Doc d) {
		double docLogLikelihood = 0;
		
		for (_Word w : d.getTestWords()) {
			int wid = w.getIndex();
			
			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double wordPerTopicLikelihood = d.m_topics[k]
						* topic_term_probabilty[k][wid];
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			
			docLogLikelihood += Math.log(wordLogLikelihood);
		}

		return docLogLikelihood;
	}
}
