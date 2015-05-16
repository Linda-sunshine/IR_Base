package topicmodels.multithreads;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.pLSA;

public class pLSA_multithread extends pLSA {

	class pLSA_workers implements Runnable {
		//int number_of_topics;
		double[][] sstat;
		ArrayList<_Doc> m_corpus;
		double m_likelihood;
		
		public pLSA_workers() {
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
		
		private double calculate_E_step(_Doc d) {	
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
		
		double[][] getStats() {
			return sstat;
		}
		
		void resetStats() {
			for(int i=0; i<sstat.length; i++)
				Arrays.fill(sstat[i], 0);
		}
	}
	
	Thread[] m_threadpool;
	pLSA_workers[] m_workers;
	
	public pLSA_multithread(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			double lambda, double back_ground [], //arguments for 2topic topic model
			int number_of_topics, double alpha) {
		super(number_of_iteration, converge, beta, c, lambda, back_ground, number_of_topics, alpha);
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
		m_workers = new pLSA_workers[cores];
		
		for(int i=0; i<cores; i++)
			m_workers[i] = new pLSA_workers();
		
		int workerID = 0;
		for(_Doc d:collection) {
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
	
	private double multithread_E_step() {
		for(int i=0; i<m_workers.length; i++) {
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(int i=0; i<m_threadpool.length; i++){
			try {
				m_threadpool[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		double[][] sstat;
		for(int i=0; i<m_workers.length; i++) {
			sstat = m_workers[i].getStats();
			for(int k=0; k<number_of_topics; k++) {
				for (int v=0; v<vocabulary_size; v++)
					word_topic_sstat[k][v] += sstat[k][v];				
			}
		}
		return 0;
	}
	
	@Override
	public void EM() {	
		long starttime = System.currentTimeMillis();
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do {
			init();
			
			current = multithread_E_step();			
			calculate_M_step(i);
			
			current += calculate_log_likelihood();//together with corpus-level log-likelihood
			if (i>0)
				delta = (last-current)/last;
			else
				delta = 1.0;
			last = current;
			
			if (m_display && i%10==0) {
				if (this.m_converge>0)
					System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
				else {
					System.out.print(".");
					if (i%200==190)
						System.out.println();
				}
			}
			
			if (Math.abs(delta)<m_converge)
				break;//to speed-up, we don't need to compute likelihood in many cases
		} while (++i<this.number_of_iteration);
		
		finalEst();
		
		long endtime = System.currentTimeMillis() - starttime;
		System.out.format("Likelihood %.3f after step %s converge to %f after %d seconds...\n", current, i, delta, endtime/1000);	
	}
}
