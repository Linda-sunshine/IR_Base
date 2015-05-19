package topicmodels;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected int number_of_iteration;
	protected double m_converge;
	protected _Corpus m_corpus;	
	
	//for training/testing split
	protected ArrayList<_Doc> m_trainSet, m_testSet;
	
	//smoothing parameter for p(w|z, \beta)
	protected double d_beta; 	
	
	protected boolean m_display; // output EM iterations
	protected boolean m_collectCorpusStats; // if we will collect corpus-level statistics (for efficiency purpose)
	
	protected boolean m_multithread = false; // by default we do not use multi-thread mode
	protected Thread[] m_threadpool = null;
	protected TopicModelWorker[] m_workers = null;
	
	public TopicModel(int number_of_iteration, double converge, double beta, _Corpus c) {
		this.vocabulary_size = c.getFeatureSize();
		this.number_of_iteration = number_of_iteration;
		this.m_converge = converge;
		this.d_beta = beta;
		this.m_corpus = c;
		
		m_display = true; // by default we will track EM iterations
	}
	
	@Override
	public String toString() {
		return "Topic Model";
	}
	
	public void setDisplay(boolean disp) {
		m_display = disp;
	}
	
	//initialize necessary model parameters
	protected abstract void initialize_probability(Collection<_Doc> collection);	
	
	// to be called per EM-iteration
	protected abstract void init();
	
	// to be called by the end of EM algorithm 
	protected abstract void finalEst();
	
	// to be call per test document
	protected abstract void initTestDoc(_Doc d);
	
	//estimate posterior distribution of p(\theta|d)
	protected abstract void estThetaInDoc(_Doc d);
	
	// perform inference of topic distribution in the document
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double delta, last = 1, current;
		int  i = 0;
		do {
			current = calculate_E_step(d);
			estThetaInDoc(d);			
			
			delta = (last - current)/last;
			last = current;
		} while (Math.abs(delta)>m_converge && ++i<this.number_of_iteration);
		return current;
	}
		
	//E-step should be per-document computation
	public abstract double calculate_E_step(_Doc d); // return log-likelihood
	
	//M-step should be per-corpus computation
	public abstract void calculate_M_step(int i); // input current iteration to control sampling based algorithm
	
	//compute per-document log-likelihood
	protected abstract double calculate_log_likelihood(_Doc d);
	
	//print top k words under each topic
	public abstract void printTopWords(int k);
	
	// compute corpus level log-likelihood
	protected double calculate_log_likelihood() {
		return 0;
	}
	
	public void EMonCorpus() {
		m_trainSet = m_corpus.getCollection();
		EM();
	}
	
	double multithread_E_step() {
		for(int i=0; i<m_workers.length; i++) {
			m_threadpool[i] = new Thread(m_workers[i]);
			m_threadpool[i].start();
		}
		
		//wait till all finished
		for(Thread thread:m_threadpool){
			try {
				thread.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		double likelihood = 0;
		for(TopicModelWorker worker:m_workers)
			likelihood += worker.accumluateStats();
		return likelihood;
	}

	public void EM() {	
		long starttime = System.currentTimeMillis();
		
		m_collectCorpusStats = true;
		initialize_probability(m_trainSet);
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do {
			init();
			
			if (m_multithread)
				current = multithread_E_step();
			else {
				current = 0;
				for(_Doc d:m_trainSet)
					current += calculate_E_step(d);
			}
			
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
		
		tmpSimCheck();
	}
	
	private int getLabel(int y) {
//		return y;
		
		//turn into binary
		if (y>=3)
			return 1;
		else
			return 0;
	}
	
	public void tmpSimCheck() {
		if (m_trainSet==null)
			m_trainSet = m_corpus.getCollection();
		
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./data/matlab/test.dat"), "UTF-8"));
			double similarity;
			int yi, yj, pairSize = 0;
			for(int i=0; i<m_trainSet.size(); i++) {				
				_Doc di = m_trainSet.get(i);
				yi = getLabel(di.getYLabel());
				
				for(int j=i+1; j<m_trainSet.size(); j++) {
					_Doc dj = m_trainSet.get(j);
					
					if (Math.random() < 0.95)//di.getItemID().equals(dj.getItemID()) == false || 
						continue;
					
					//if we have topics
					similarity = Utils.KLsymmetric(di.m_topics, dj.m_topics) / di.m_topics.length;
					
					//if we only have bag-of-words
//					similarity = Utils.calculateSimilarity(di, dj);					
					
					yj = getLabel(dj.getYLabel());
					writer.write(String.format("%s %.5f\n", yi==yj, similarity));	
					pairSize ++;
				}
			}
			writer.close();
			System.out.format("%d pairs generated for verification purpose.\n", pairSize);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public double Evaluation() {
		m_collectCorpusStats = false;
		double perplexity = 0, loglikelihood, log2 = Math.log(2.0), sumLikelihood = 0;
		for(_Doc d:m_testSet) {
			loglikelihood = inference(d);
			sumLikelihood += loglikelihood;
			perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
		}
		perplexity /= m_testSet.size();
		sumLikelihood /= m_testSet.size();
		
		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		return perplexity;
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k) {
		m_corpus.shuffle(k);
		int[] masks = m_corpus.getMasks();
		ArrayList<_Doc> docs = m_corpus.getCollection();
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		double[] perf = new double[k];
		
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				if( masks[j]==i ) 
					m_testSet.add(docs.get(j));
				else 
					m_trainSet.add(docs.get(j));
			}
			
			long start = System.currentTimeMillis();
			EM();
			perf[i] = Evaluation();
			System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			m_trainSet.clear();
			m_testSet.clear();
		}
		
		//output the performance statistics
		double mean = Utils.sumOfArray(perf)/k, var = 0;
		for(int i=0; i<perf.length; i++)
			var += (perf[i]-mean) * (perf[i]-mean);
		var = Math.sqrt(var/k);
		System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
	}
}
