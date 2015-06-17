package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LDA_Variational;
import topicmodels.LRHTMM;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.jsonAnalyzer;

public class TopicModelMain {

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "HTSM"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM
		
		int number_of_topics = 20;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int topK = 20, number_of_iteration = 100, crossV = 1;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM"))
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
		String aspectlist = "./data/Model/aspect_tablet.txt";

		/*****Parameters in feature selection.*****/
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.5; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		
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
			pLSA model = new pLSA_multithread(number_of_iteration, converge, beta, c, 
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha);
			
			model.setDisplay(true);
			model.LoadPrior(aspectlist, eta);
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("LDA_Gibbs")) {		
			LDA_Gibbs model = new LDA_Gibbs(number_of_iteration, converge, beta, c, 
				lambda, analyzer.getBackgroundProb(), 
				number_of_topics, alpha, 0.4, 50);
		
			model.setDisplay(true);
			model.LoadPrior(aspectlist, eta);
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		}  else if (topicmodel.equals("LDA_Variational")) {		
			LDA_Variational model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha, 10, -1);
			
				model.setDisplay(true);
				model.LoadPrior(aspectlist, eta);
				if (crossV<=1) {
					model.EMonCorpus();
					model.printTopWords(topK);
				} else 
					model.crossValidation(crossV);
		}  else if (topicmodel.equals("HTMM")) {
			HTMM model = new HTMM(number_of_iteration, converge, beta, c, 
					number_of_topics, alpha);
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("HTSM")) {
			HTSM model = new HTSM(number_of_iteration, converge, beta, c, 
					number_of_topics, alpha);
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		}  
		else if (topicmodel.equals("LRHTMM")) {
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
