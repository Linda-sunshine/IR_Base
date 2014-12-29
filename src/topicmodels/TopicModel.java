package topicmodels;

import java.io.IOException;
import java.text.ParseException;

import Analyzer.jsonAnalyzer;
import structures._Corpus;
import structures._Doc;

public abstract class TopicModel {
	protected int number_of_topics;
	protected int vocabulary_size;
	protected int number_of_iteration;
	protected _Corpus m_corpus;
	
	//smoothing parameter for p(w|z, \theta)
	protected double d_beta; 	
	
	//initialize necessary model parameters
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
	
	//print top k words under each topic
	public abstract void printTopWords(int k);
	
	// perform inference of topic distribution in the document
	public abstract double[] get_topic_probability(_Doc d);
	
	// to be called per EM-iteration
	protected abstract void init();
	
	public TopicModel(int number_of_iteration, double beta, _Corpus c) {
		vocabulary_size = c.getFeatureSize();
		this.number_of_iteration = number_of_iteration;
		this.d_beta = beta;
		this.m_corpus = c;
	}
	
	public void EM(double converge)
	{	
		initialize_probability();
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			init();
			for(_Doc d:m_corpus.getCollection())
				calculate_E_step(d);
			
			calculate_M_step();
			
			current = calculate_log_likelihood();
			delta = Math.abs((current - last)/last);
			last = current;
			
			System.out.format("Likelihood %.3f at step %s converge to %f...\n", current, i, delta);
			i++;
			
		} while (delta>converge && i<this.number_of_iteration);
	}
	
	public static void main(String[] args) throws IOException, ParseException
	{	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "HTMM"; // 2topic, pLSA, HTMM
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3;//these two parameters must be larger than 1!!!
		double converge = 1e-5, lambda = 0.7;
		int topK = 10, number_of_iteration = 50;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		if (topicmodel.equals("HTMM"))
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
		jsonAnalyzer 
		analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold, stnModel);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		
		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, lambda, beta, analyzer.get_back_ground_probabilty(), c);
			
			for(_Doc d:c.getCollection()) {
				model.get_topic_probability(d);
				model.printTopWords(topK);
			}
		} else if (topicmodel.equals("pLSA")) {
			pLSA model = new pLSA(number_of_topics, number_of_iteration, lambda,  beta, alpha, analyzer.get_back_ground_probabilty(),c);
			
			model.EM(converge);
			model.printTopWords(topK);
		} else if (topicmodel.equals("HTMM")) {
			HTMM model = new HTMM(number_of_topics, alpha, beta, number_of_iteration, c);
			
			model.EM(converge);
			model.printTopWords(topK);
		}
	}
}
