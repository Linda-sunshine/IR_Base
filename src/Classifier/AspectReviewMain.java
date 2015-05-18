package Classifier;

import influence.PageRank;
import java.io.IOException;
import java.text.ParseException;
import structures._Corpus;
import topicmodels.LDA_Gibbs;
import topicmodels.pLSA;
import Analyzer.AspectAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class AspectReviewMain {
	
	/**We use AmazonReviewMain.java to generate the features. 
	 * Then, we use bootstrapping to generate aspect keywords.
	 * Finally, we use the generated keywords for aspects to do GF-RW.
	 **/
	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold

		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
	
		//"SUP", "SEMI", "FV", "ASPECT"
		String style = "SEMI";
		
		//"NB", "LR", "SVM", "PR"
		String classifier = "GF-RW"; //Which classifier to use.
		String multipleLearner = "SVM";
		double C = 1;
		String debugOutput = "data/debug/Debug_" + classifier + ".output";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);
		
		/*****The parameters used in loading files.*****/
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Token model.
		String aspectModel = "./data/Model/aspectlist.txt"; // list of keywords in each aspect
		String aspectInput = "./data/Model/aspect_input.txt";
		String aspectOutput = "./data/Model/aspect_output_0515.txt"; // list of keywords in each aspect
		
		String stopwords = "./data/Model/stopwords.dat";
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		
		String featureSelection = "DF"; //Feature selection method.
		String pattern = String.format("%dgram_%s_%s", Ngram, featureValue, featureSelection);
		String fvFile = String.format("data/Features/fv_%s.txt", pattern);
		String fvStatFile = String.format("data/Features/fv_stat_%s.txt", pattern);
		String vctFile = String.format("data/Fvs/vct_%s.dat", pattern);		
		
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1.0; // Used in feature selection, the ending point of the features.
		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
		int chiSize = 50; // top ChiSquare words for aspect keyword selection

		/*****Parameters in time series analysis.***/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
//		System.out.println("Performing feature selection, wait...");
//		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, "", Ngram, lengthThreshold, aspectModel, chiSize);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
	
//		/****Load json files to do Aspect annotation*****/
//		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold, aspectInput, chiSize);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.BootStrapping(aspectOutput, 0.9, 10);		
			
//		/****create vectors for documents*****/
//		System.out.println("Creating feature vectors, wait...");
//		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, aspectOutput, chiSize);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.setFeatureValues(featureValue, norm);
//		analyzer.setTimeFeatures(window);
			
		/****create vectors for documents*****/
		boolean topicFlag = true;
		System.out.println("Creating feature vectors, wait...");
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, aspectOutput, window, topicFlag);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		
		featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.getCorpus();
		System.out.println("The number of reviews with non-zero apsects: " + analyzer.returnCount());
		
		/***The parameters used in GF-RW.****/
		double eta_rw = 0.2;
		double sr = 1;
			
		/***Try LDA_Gibbs****/
		if(topicFlag){
			int number_of_topics = 30;
			double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta_lad = 5.0;//these two parameters must be larger than 1!!!
			double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
			int topK = 10, number_of_iteration = 100;
			
			pLSA model = new pLSA(number_of_iteration, converge, beta, corpus, lambda, analyzer.getBackgroundProb(), number_of_topics, alpha);
//			LDA_Gibbs mode = new LDA_Gibbs(number_of_iteration, converge, beta, corpus, lambda, analyzer.getBackgroundProb(), number_of_topics, alpha, 0.4, 50);
			model.setDisplay(true);
			model.LoadPrior(fvFile, aspectOutput, eta_lad);
			model.EMonCorpus();
			model.printTopWords(topK);
		}
		
		//temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
		featureValue = "BM25";//Change the feature value for SVM.
		analyzer.setFeatureValues(featureValue, norm);
				
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize + window + 1);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
					
			} else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize + window + 1, C);
				myLR.setDebugOutput(debugOutput);
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize + window + 1, C, 0.01);//default eps value from Lin's implementation
				mySVM.crossValidation(CVFold, corpus);
					
			} else if (classifier.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize + window + 1, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
					
			} else System.out.println("Classifier has not developed yet!");
		} else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize, multipleLearner);
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW")) {
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, sr, 40, 20, 1, 0.1, 1e-4, eta_rw, false);
				mySemi.setFeaturesLookup(analyzer.getFeaturesLookup()); //give the look up to the classifier for debugging purpose.
				mySemi.setTopicFlag(true);
				mySemi.setDebugOutput(debugOutput);//For debug purpose.
//				mySemi.setDebugPrinters(WrongRWfile, WrongSVMfile, FuSVM);
//				mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile));
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW-ML")) {
				LinearSVMMetricLearning lMetricLearner = new LinearSVMMetricLearning(corpus, classNumber, featureSize, multipleLearner, 0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false, 3, 0.01);
				lMetricLearner.setDebugOutput(debugOutput);
				lMetricLearner.crossValidation(CVFold, corpus);
			} else System.out.println("Classifier has not been developed yet!");
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else 
			System.out.println("Learning paradigm has not developed yet!");
	}
}
