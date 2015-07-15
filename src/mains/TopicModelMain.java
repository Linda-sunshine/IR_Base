package mains;

import java.io.IOException;
import java.text.ParseException;
import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LRHTMM;
import topicmodels.LRHTSM;
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
		String topicmodel = "LDA_Variational"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 10, number_of_iteration = 50, crossV = 5;
		int gibbs_iteration = 1000, gibbs_lag = 50;
		double burnIn = 0.4;
		boolean display = true;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablet/topicmodel";
		//String folder = "./data/amazon/test";
		//String folder = "./data/amazon/newegg/newegg-reviews.json";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
		}
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
		String aspectlist = null;
		//String aspectlist = "./data/Model/aspect_tablet.txt";
		//String aspectlist = "./data/Model/aspect_sentiment_tablet.txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		
		/*****Parameters in feature selection.*****/
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.5; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.
		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
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
		} else if (topicmodel.equals("Tensor")) {
			c.saveAs3WayTensor("./data/vectors/3way_tensor.txt");
		} else {
			pLSA model = null;
			
			if (topicmodel.equals("pLSA")) {
				model = new pLSA_multithread(number_of_iteration, converge, beta, c, 
						lambda, analyzer.getBackgroundProb(), 
						number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {		
				model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, //in gibbs sampling, no need to compute log-likelihood during sampling
					lambda, analyzer.getBackgroundProb(), 
					number_of_topics, alpha, burnIn, gibbs_lag);
			}  else if (topicmodel.equals("LDA_Variational")) {		
				model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, analyzer.getBackgroundProb(), 
						number_of_topics, alpha, varIter, varConverge);
			}  else if (topicmodel.equals("HTMM")) {
				model = new HTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("HTSM")) {
				model = new HTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			}  
			else if (topicmodel.equals("LRHTMM")) {
				c.setStnFeatures();				
				model = new LRHTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			}
			else if (topicmodel.equals("LRHTSM")) {
				c.setStnFeatures();
				c.setStnFeaturesForSentiment(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
				model = new LRHTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			}
			
			model.setDisplay(display);
			model.LoadPrior(aspectlist, eta);
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else 
				model.crossValidation(crossV);
		}
	}
}
