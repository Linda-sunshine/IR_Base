package topicmodels;

import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected double m_converge;//relative change in log-likelihood to terminate EM
	protected int number_of_iteration;//number of iterations in inferencing testing document
	protected _Corpus m_corpus;	
	protected int m_crossValidFold;
	
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
	public abstract void printTopWords(int k, boolean logSpace);
	
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
		
		if(this instanceof HTSM)
			calculatePrecisionRecall();
		System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, sumLikelihood);
		return perplexity;
	}
	
	public void calculatePrecisionRecall(){
		int[][] precision_recall = new int [2][2];
		precision_recall [0][0] = 0; // 0 is for pos
		precision_recall[0][1] = 0; // 1 is neg 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;
		
		int actualLabel, predictedLabel;
		
		for(_Doc d:m_testSet) {
			// if documnet is from newEgg which is 2 then calculate precision-recall
			if(d.getSourceName()==2){
				
				for(int i=0; i<d.getSenetenceSize(); i++){
					actualLabel = d.getSentence(i).getSentenceSenitmentLabel();
					predictedLabel = d.getSentence(i).getSentencePredictedSenitmentLabel();
					precision_recall[actualLabel][predictedLabel]++;
				}
			}
		}
		
		System.out.println("Confusion Matrix");
		for(int i=0; i<2; i++)
		{
			for(int j=0; j<2; j++)
			{
				System.out.print(precision_recall[i][j]+",");
			}
			System.out.println();
		}
		
		double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
		double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
		
		
		double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
		double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
		
		System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
		
		
		double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
		double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
		
		System.out.println("F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k) {
		m_crossValidFold = k;
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
