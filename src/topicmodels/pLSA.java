package topicmodels;


/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * Probabilistic Latent Semantic Analysis Topic Modeling 
 */

import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;


public class pLSA extends twoTopic {
	// Dirichlet prior for p(\theta|d)
	double d_alpha; // smoothing of p(z|d)
	
	double[][] topic_term_probabilty ; /* p(w|z) */
	double[][] word_topic_sstat; /* fractional count for p(z|d,w) */
	
	public pLSA(int number_of_topics, int number_of_iteration, double lambda, double beta, double alpha,
			double back_ground [], _Corpus c) {			
		super(number_of_iteration, lambda, beta, back_ground, c);
		
		this.d_alpha = alpha;
		this.number_of_topics = number_of_topics;
		topic_term_probabilty = new double[this.number_of_topics][this.vocabulary_size];
		word_topic_sstat = new double[this.number_of_topics][this.vocabulary_size];
	}

	@Override
	protected void initialize_probability()
	{	
		// initialize topic document proportion, p(z|d)
		for(_Doc d:m_corpus.getCollection())
			d.setTopics(number_of_topics, d_alpha);//allocate memory and randomize it
		
		// initialize term topic matrix p(w|z,\phi)
		for(int i=0;i<number_of_topics;i++)
			Utils.randomize(this.topic_term_probabilty[i], d_beta);
	}
	
	@Override
	public void calculate_E_step(_Doc d)
	{	
		double propB; // background proportion
		double exp; // expectation of each term under topic assignment
		for(_SparseFeature fv:d.getSparse()) {
			int j = fv.getIndex(); // jth word in doc
			
			//-----------------compute posterior----------- 
			double sum = 0;
			for(int k=0;k<this.number_of_topics;k++)
				sum += d.m_topics[k]*topic_term_probabilty[k][j];//shall we computate it in log space?
			
			propB = m_lambda * background_probability[j];
			propB /= propB + (1-m_lambda) * sum;//posterior of background probability
			
			//-----------------compute and accumulate expectations----------- 
			for(int k=0;k<this.number_of_topics;k++) {
				exp = fv.getValue() * (1-propB)*d.m_topics[k]*topic_term_probabilty[k][j]/sum;
				word_topic_sstat[k][j] += exp;
				d.m_sstat[k] += exp;
			}
		}
	}
	
	@Override
	public void calculate_M_step()
	{	
		// update topic-term matrix -------------
		double sum = 0;
		for(int k=0;k<this.number_of_topics;k++) {
			sum = Utils.sumOfArray(word_topic_sstat[k]); // smoothing
			for(int i=0;i<this.vocabulary_size;i++)
				topic_term_probabilty[k][i] = (word_topic_sstat[k][i] + d_beta) / sum;
		}
		
		// update per-document topic distribution vectors
		for(_Doc d:m_corpus.getCollection()) {
			sum = Utils.sumOfArray(d.m_sstat) + number_of_topics*d_alpha;
			for(int k=0;k<this.number_of_topics;k++)
				d.m_topics[k] = (d.m_sstat[k]+d_alpha) / sum; // smoothing
		}
	}
	
	@Override
	protected void init() { // clear up for next iteration
		for(int k=0;k<this.number_of_topics;k++)
			Arrays.fill(word_topic_sstat[k], d_beta-1.0);
		
		for(_Doc d:m_corpus.getCollection())
			Arrays.fill(d.m_sstat, d_alpha-1.0);
	}
	
	//pLSA cannot directly infer topic proportion for new documents!
	@Override
	public double[] get_topic_probability(_Doc d) {
		return d.m_topics;
	}
	
	/*likelihod calculation */
	/* M is number of doc
	 * N is number of word in corpus
	 */
	/* p(w,d) = sum_1_M sum_1_N count(d_i, w_j) * log[ lambda*p(w|theta_B) + [lambda * sum_1_k (p(w|z) * p(z|d)) */ 
	@Override
	public double calculate_log_likelihood(_Doc d)
	{
		double logLikelihood = 0.0, prob;
		for(_SparseFeature fv:d.getSparse()) {
			int j = fv.getIndex();	
			prob = 0.0;
			for(int k=0;k<this.number_of_topics;k++)//\sum_z p(w|z,\theta)p(z|d)
				prob += d.m_topics[k]*topic_term_probabilty[k][j];
			prob = prob*(1-m_lambda) + this.background_probability[j]*m_lambda;//(1-\lambda)p(w|d) * \lambda p(w|theta_b)
			logLikelihood += fv.getValue() * Math.log(prob);
		}
		return logLikelihood;
	}
	
	@Override
	public void printTopWords(int k) {
		for(int i=0; i<topic_term_probabilty.length; i++) {
			MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
			for(int j = 0; j < vocabulary_size; j++)
				fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));
			for(_RankItem it:fVector)
				System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
			System.out.println();
		}
	}
}
