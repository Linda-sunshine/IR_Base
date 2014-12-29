package topicmodels;

import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * two-topic Topic Modeling 
 */

public class twoTopic extends TopicModel {
	private double[] m_theta;//p(w|\theta) - the only topic for each document
	private double[] m_sstat;//c(w,d)p(z|w) - sufficient statistics for each word under topic
	/*p (w|theta_b) */
	protected double[] background_probability;
	protected double m_lambda; //proportion of background topic in each document
	
	public twoTopic(int number_of_iteration, double lambda, double beta, double[] back_ground, _Corpus c) {
		super(number_of_iteration, beta, c);
		
		background_probability = back_ground;
		m_lambda = lambda;
		m_theta = new double[vocabulary_size];
		m_sstat = new double[vocabulary_size];
	}
	
	@Override
	protected void initialize_probability() {	
    	Utils.randomize(m_theta, d_beta);
    	Arrays.fill(m_sstat, 0);
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		for(_SparseFeature fv:d.getSparse()) {
			int wid = fv.getIndex();
			m_sstat[wid] = (1-m_lambda)*m_theta[wid];
			m_sstat[wid] = fv.getValue() * m_sstat[wid]/(m_sstat[wid]+m_lambda*background_probability[wid]);//compute the expectation
		}
	}
	
	@Override
	public void calculate_M_step()
	{		
		double sum = Utils.sumOfArray(m_sstat) + vocabulary_size * d_beta;//with smoothing
		for(int i=0;i<vocabulary_size;i++)
			m_theta[i] = (d_beta+m_sstat[i]) / sum;
	}
	
	protected double calculate_log_likelihood(_Doc d)
	{		
		double logLikelihood = 0.0;
		for(_SparseFeature fv:d.getSparse())
		{
			int wid = fv.getIndex();
			logLikelihood += fv.getValue() * Math.log(m_lambda*background_probability[wid] + (1-m_lambda)*m_theta[wid]);
		}
		
		return logLikelihood;
	}

	@Override
	public void printTopWords(int k) {
		//we only have one topic to show
		MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
		for(int i=0; i<m_theta.length; i++) 
			fVector.add(new _RankItem(m_corpus.getFeature(i), m_theta[i]));
		
		for(_RankItem it:fVector)
			System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
		System.out.println();
	}
	
	//this is mini-EM in a single document 
	@Override
	public double[] get_topic_probability(_Doc d)
	{
		initialize_probability();
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			calculate_E_step(d);
			calculate_M_step();
			
			current = calculate_log_likelihood(d);
			delta = Math.abs((current - last)/last);
			last = current;
			i++;
		} while (delta>1e-4 && i<this.number_of_iteration);
		
		double perplexity = Math.exp(-current/d.getTotalDocLength());
		System.out.format("Likelihood in document %s converges to %.4f after %d steps...\n", d.getName(), perplexity, i);
		return m_theta;
	}
	
	protected void init() {};
}
