package topicmodels.multithreads;

import java.util.ArrayList;
import java.util.Arrays;

import structures._Doc;

//E-step needs to be implemented by different models
public abstract class TopicModel_worker implements TopicModelWorker {

	public enum RunType {
		RT_inference,
		RT_EM
	}
	
	protected ArrayList<_Doc> m_corpus;
	protected double m_likelihood;
	protected double m_perplexity;
	
	double[][] sstat; // p(w|z)
	int number_of_topics;
	int vocabulary_size;
	RunType m_type = RunType.RT_EM;//EM is the default type 	
	
	public TopicModel_worker(int number_of_topics, int vocabulary_size) {
		this.number_of_topics = number_of_topics;
		this.vocabulary_size = vocabulary_size;
		
		m_corpus = new ArrayList<_Doc>();
		sstat = new double[number_of_topics][vocabulary_size];
	}
	
	public void setType(RunType type) {
		m_type = type;
	}
	
	@Override
	public double getLogLikelihood() {
		return m_likelihood;
	}
	
	@Override
	public double getPerplexity() {
		return m_perplexity;
	}
	
	@Override
	public void run() {
		m_likelihood = 0;
		m_perplexity = 0;
		
		double loglikelihood = 0, log2 = Math.log(2.0);
		for(_Doc d:m_corpus) {
			if (m_type == RunType.RT_EM)
				m_likelihood += calculate_E_step(d);
			else if (m_type == RunType.RT_inference) {
				loglikelihood = inference(d);
				m_perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
				m_likelihood += loglikelihood;
			}
		}
	}
	
	@Override
	public void addDoc(_Doc d) {
		m_corpus.add(d);
	}
	
	public void clearCorpus() {
		m_corpus.clear();
	}
	
	@Override
	public void resetStats() {
		for(int i=0; i<sstat.length; i++)
			Arrays.fill(sstat[i], 0);			
	}

	@Override
	public double accumluateStats(double[][] word_topic_sstat) {
		for(int k=0; k<number_of_topics; k++) {
			for(int v=0; v<vocabulary_size; v++)
				word_topic_sstat[k][v] += sstat[k][v];
		}
		return m_likelihood;
	}	
}
