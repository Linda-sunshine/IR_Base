package topicmodels.multithreads;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.pLSA;

public class pLSA_multithread extends pLSA {

	class pLSA_worker implements TopicModelWorker {
		//int number_of_topics;
		double[][] sstat;
		ArrayList<_Doc> m_corpus;
		double m_likelihood;
		
		public pLSA_worker() {
			sstat = new double[number_of_topics][vocabulary_size];
			m_corpus = new ArrayList<_Doc>();
		}
		
		public void addDoc(_Doc d) {
			m_corpus.add(d);
		}
		
		@Override
		public void run() {
			m_likelihood = 0;
			for(_Doc d:m_corpus)
				m_likelihood += calculate_E_step(d);
		}
		
		public double calculate_E_step(_Doc d) {	
			double propB; // background proportion
			double exp; // expectation of each term under topic assignment
			for(_SparseFeature fv:d.getSparse()) {
				int j = fv.getIndex(); // jth word in doc
				double v = fv.getValue();
				
				//-----------------compute posterior----------- 
				double sum = 0;
				for(int k=0;k<number_of_topics;k++)
					sum += d.m_topics[k]*topic_term_probabilty[k][j];//shall we compute it in log space?
				
				propB = m_lambda * background_probability[j];
				propB /= propB + (1-m_lambda) * sum;//posterior of background probability
				
				//-----------------compute and accumulate expectations----------- 
				for(int k=0;k<number_of_topics;k++) {
					exp = v * (1-propB)*d.m_topics[k]*topic_term_probabilty[k][j]/sum;
					d.m_sstat[k] += exp;
					
					sstat[k][j] += exp;
				}
			}
			
			return calculate_log_likelihood(d);
		}
		
		public double accumluateStats() {
			for(int k=0; k<number_of_topics; k++) 
				for (int v=0; v<vocabulary_size; v++)
					word_topic_sstat[k][v] += sstat[k][v];
			return m_likelihood;
		}
		
		public void resetStats() {
			for(int i=0; i<sstat.length; i++)
				Arrays.fill(sstat[i], 0);
		}
	}
	
	public pLSA_multithread(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			double lambda, double back_ground [], //arguments for 2topic topic model
			int number_of_topics, double alpha) {
		super(number_of_iteration, converge, beta, c, lambda, back_ground, number_of_topics, alpha);
		m_multithread = true;
	}
	
	@Override
	public String toString() {
		return String.format("multi-thread pLSA[k:%d, lambda:%.2f]", number_of_topics, m_lambda);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);
		
		int cores = Runtime.getRuntime().availableProcessors();
		m_threadpool = new Thread[cores];
		m_workers = new pLSA_worker[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new pLSA_worker();
		
		int workerID = 0;
		for(_Doc d:collection) {//evenly allocate the work load
			m_workers[workerID%cores].addDoc(d);
			workerID++;
		}
	}
	
	@Override
	protected void init() { // clear up for next iteration
		super.init();
		for(int i=0; i<m_workers.length; i++)
			m_workers[i].resetStats();
	}
}
