package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LRHTMM;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.LRHTSM_multithread;
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
		String topicmodel = "pLSA"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		
		String category = "tablet";
		int number_of_topics = 10;
		boolean loadNewEggInTrain = false; // false means in training there is no reviews from new
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int testDocMod = 11; // when setRandomFold = false, we select every m_testDocMod_th document for testing
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 10, number_of_iteration = 50, crossV = 1;
		int gibbs_iteration = 1500, gibbs_lag = 50;
		double burnIn = 0.4;
		boolean display = true, sentence = false;
		
		// most popular items under each category from Amazon
		// needed for docSummary
		String tabletProductList[] = {"B00G3Q4CMM"};
		String cameraProductList[] = {"B00FY3U206"};
		String phoneProductList[] = {"B00H0MGCDK"};
		String tvProductList[] = {"B00GEECXKQ"};
		
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
			sentence = true;
		}
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
		String aspectlist = null;
	
		String aspectList = "./data/Model/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";
		
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
						lambda, number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {		
				model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, //in gibbs sampling, no need to compute log-likelihood during sampling
					lambda, number_of_topics, alpha, burnIn, gibbs_lag);
			} else if (topicmodel.equals("LDA_Variational")) {		
				model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha, varIter, varConverge);
			} else if (topicmodel.equals("HTMM")) {
				model = new HTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("HTSM")) {
				model = new HTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("LRHTMM")) {
				c.setStnFeatures();				
				model = new LRHTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			} else if (topicmodel.equals("LRHTSM")) {
				c.setStnFeatures();
				c.setStnFeaturesForSentiment(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
				model = new LRHTSM_multithread(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			}
			
			model.setDisplay(display);
			model.setNewEggLoadInTrain(loadNewEggInTrain);
			
			if(loadAspectSentiPrior==1){
				System.out.println("Loading Ascpect Senti list from "+aspectSentiList);
				model.LoadPrior(aspectSentiList, eta);
			} else if(loadAspectSentiPrior==2){
				System.out.println("Loading Ascpect list from "+aspectList);
				model.LoadPrior(aspectList, eta);
			}else{
				System.out.println("No prior is added!!");
			}
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else {
				model.setTestDocMod(testDocMod);
				model.setRandomFold(setRandomFold);
				model.crossValidation(crossV);
				
				if (sentence) {
					if(category.equalsIgnoreCase("camera"))
						model.docSummary(cameraProductList);
					else if(category.equalsIgnoreCase("tablet"))
						model.docSummary(tabletProductList);
					else if(category.equalsIgnoreCase("phone"))
						model.docSummary(phoneProductList);
					else if(category.equalsIgnoreCase("tv"))
						model.docSummary(tvProductList);
				}
			}
			
		}
	}
}
