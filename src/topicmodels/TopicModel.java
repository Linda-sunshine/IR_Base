package topicmodels;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import utils.Utils;
import Analyzer.jsonAnalyzer;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected int number_of_iteration;
	protected double m_converge;
	protected _Corpus m_corpus;	
	
	//for training/testing split
	ArrayList<_Doc> m_trainSet, m_testSet;
	
	//smoothing parameter for p(w|z, \beta)
	protected double d_beta; 	
	
	boolean m_display; // output EM iterations
	
	public TopicModel(int number_of_iteration, double converge, double beta, _Corpus c) {
		this.vocabulary_size = c.getFeatureSize();
		this.number_of_iteration = number_of_iteration;
		this.m_converge = converge;
		this.d_beta = beta;
		this.m_corpus = c;
		
		m_display = false; // by default we won't track EM iterations
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
	
	protected abstract void initStatInDoc(_Doc d);
	
	// to be call per test document
	protected abstract void initTestDoc(_Doc d);
	
	//estimate posterior distribution of p(\theta|d)
	protected abstract void estThetaInDoc(_Doc d);
	
	// perform inference of topic distribution in the document
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do {
			calculate_E_step(d);
			estThetaInDoc(d);
			
			current = calculate_log_likelihood(d);
			delta = (last - current)/last;
			last = current;
		} while (Math.abs(delta)>m_converge && ++i<this.number_of_iteration);
		return current;
	}
		
	//E-step should be per-document computation
	public abstract void calculate_E_step(_Doc d);
	
	//M-step should be per-corpus computation
	public abstract void calculate_M_step();
	
	//compute per-document log-likelihood
	protected abstract double calculate_log_likelihood(_Doc d);
	
	//print top k words under each topic
	public abstract void printTopWords(int k);
	
	// compute corpus level log-likelihood
	protected double calculate_log_likelihood() {
		double logLikelihood = 0;
		for(_Doc d:m_trainSet)
			logLikelihood += calculate_log_likelihood(d);
		return logLikelihood; 
	}
	
	public void EMonCorpus() {
		m_trainSet = m_corpus.getCollection();
		EM();
	}

	public void EM() {	
		initialize_probability(m_trainSet);
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			init();
			for(_Doc d:m_trainSet)
				calculate_E_step(d);
			
			calculate_M_step();
			
			current = calculate_log_likelihood();
			delta = (last-current)/last;
			last = current;
			
			if (m_display)
				System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
			i++;
			
		} while (Math.abs(delta)>this.m_converge && i<this.number_of_iteration);
		
		if (!m_display) // output the summary
			System.out.format("Likelihood %.3f after step %s converge to %f...\n", current, i, delta);
	}
	
	public double Evaluation() {
		double perplexity = 0, loglikelihood, log2 = Math.log(2.0);
		for(_Doc d:m_testSet) {
			loglikelihood = inference(d);
			perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
		}
		perplexity /= m_testSet.size();
		System.out.format("Test set perplexity is %.3f\n", perplexity);
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
	
	public static void main(String[] args) throws IOException, ParseException
	{	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "LRHTMM"; // 2topic, pLSA, HTMM, LRHTMM, Tensor
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3;//these two parameters must be larger than 1!!!
		double converge = 1e-4, lambda = 0.7;
		int topK = 10, number_of_iteration = 100, crossV = 5;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM"))
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		
		String featureLocation = "./data/Features/selected_fv_topicmodel.txt";
		String finalLocation = "./data/Features/selected_fv_stat_topicmodel.txt";

		/*****Parameters in feature selection.*****/
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.3; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold, stnModel);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		
		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, converge, beta, c, lambda, analyzer.getBackgroundProb());
			
			if (crossV<=1) {
				for(_Doc d:c.getCollection()) {
					model.inference(d);
					model.printTopWords(topK);
				}
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("pLSA")) {			
			pLSA model = new pLSA(number_of_iteration, converge, beta, c, 
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha);
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("HTMM")) {
			HTMM model = new HTMM(number_of_iteration, converge, beta, c, 
					number_of_topics, alpha);
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("LRHTMM")) {
			c.setStnFeatures();
			
			LRHTMM model = new LRHTMM(number_of_iteration, converge, beta, c, 
					number_of_topics, alpha,
					lambda);
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("Tensor")) {
			c.saveAs3WayTensor("./data/vectors/3way_tensor.txt");
		}
		
	}
}
