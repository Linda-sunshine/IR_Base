package topicmodels;

import structures._Corpus;
import structures._Doc;

public abstract class TopicModel {
	
	protected int vocabulary_size;
	protected int number_of_iteration;//maximum number of iterations
	protected double lambda;//proportion of background topic in each document
	protected double beta; // smoothing parameter for p(w|\theta)
	
	_Corpus m_corpus;
	
	/*p (w|theta_b) */
	protected double background_probability [];
	protected abstract void initialize_probability();	
	
	//E-step should be per-document computation
	public abstract void calculate_E_step(_Doc d);
	
	//M-step should be per-corpus computation
	public abstract void calculate_M_step();
	
	//compute per-document log-likelihood
	protected abstract double calculate_log_likelihood(_Doc d);
	
	// compute corpus level log-likelihood
	protected double calculate_log_likelihood() {
		double logLikelihood = 0;
		for(_Doc d:m_corpus.getCollection())
			logLikelihood += calculate_log_likelihood(d);
		return logLikelihood;
	}
	
	public abstract double[] get_topic_probability(_Doc d);
	
	public TopicModel(int vSize, int iteration, double lambda, double beta) {
		vocabulary_size = vSize;
		number_of_iteration = iteration;
		this.lambda = lambda;
		this.beta = beta;
	}
	
	public void EM()
	{	
		initialize_probability();
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			for(_Doc d:m_corpus.getCollection())
				calculate_E_step(d);
			calculate_M_step();
			
			current = calculate_log_likelihood();
			delta = Math.abs((current - last)/last);
			current = last;
			System.out.format("Likelihood %.4f at step %s converge to %.3f...\n", current, i, delta);
			i++;
		} while (delta>1e-6 && i<this.number_of_iteration);
	}
}
