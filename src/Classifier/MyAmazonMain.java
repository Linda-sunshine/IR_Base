package Classifier;

import influence.PageRank;
import java.io.IOException;
import java.text.ParseException;
import structures._Corpus;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.KNN;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class MyAmazonMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before running the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold

		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
		
		//"SUP", "SEMI", "FV: save features and vectors to files"
		String style = "SUP";//"SUP", "SEMI"
		//Supervised: "NB", "LR", "PR-LR", "SVM"; Semi-supervised: "GF", "GF-RW", "GF-RW-ML"**/
		String classifier = "SVM"; //Which classifier to use.
		String multipleLearner = "SVM";
		double C = 1.0;
		
//		String modelPath = "./data/Model/";
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);
		
		/*****Parameters in feature selection.*****/
		String featureSelection = "DF"; //Feature selection method.
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1.0; // Used in feature selection, the ending point of the features.
		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model
		String aspectModel = "./data/Model/aspectlist.txt"; // list of keywords in each aspect
		String aspectOutput = "./data/Model/aspect_output.txt"; // list of keywords in each aspect
		
		String pattern = String.format("%dgram_%s_%s", Ngram, featureValue, featureSelection);
		String fvFile = String.format("data/Features/fv_%s_small.txt", pattern);
		String fvStatFile = String.format("data/Features/fv_stat_%s_small.txt", pattern);
		String vctFile = String.format("data/Fvs/vct_%s_small.dat", pattern);		
		
		String matrixFile = "data/metric/matrixA.dat";
		
		/***The parameters used in GF-RW and debugging.****/
		double eta = 0.1, sr = 1;
		String debugOutput = "data/debug/Debug_" + classifier + ".output";
		String WrongRWfile= "data/debug/Debug_"  + classifier + eta + "_WrongRW.txt";
		String WrongSVMfile= "data/debug/Debug_"  + classifier + eta + "_WrongSVM.txt";
		String FuSVM = "data/debug/Debug_"  + classifier + eta + "_FuSVMResults.txt";
		String reviewStatFile = "data/debug/Debug_"  + classifier + eta + "_reviewStat.txt";
	
		/**Parameters in KNN.**/
		int k = 1, l = 2;//l > 0, random projection; else brute force.
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Feature selection*****/
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
//		analyzer.resetStopwords();

		/****Create feature vectors*****/
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);		
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		//analyzer.SaveCVStat(fvStatFile);

		_Corpus corpus = analyzer.getCorpus();
		featureSize = analyzer.getFeatureSize();
//		corpus.save2File(vctFile);

//		//String matrixFile = path + "matrixA0321.dat";
//		/***Print the matrix of X and Y for metric learning.***/
//		String xFile = path + diffFolder + "X.csv";
//		String yFile = path + diffFolder + "Y.csv";
//		analyzer.printXY(xFile, yFile);
		
//		//temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
		/********Choose different classification methods.*********/
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize, C);
				myLR.setDebugOutput(debugOutput);//Save debug information into file.
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				//Define a new SVM with the parameters.
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize, C, 0.01);//default eps value from Lin's implementation
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				//Define a new Pagerank with parameters.
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, classNumber, featureSize, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else if(classifier.equals("KNN")){
				System.out.println(String.format("Start KNN, k=%d, l=%d, wait...\n", k, l));
				KNN myKNN = new KNN(corpus, classNumber, featureSize, k, l);
				myKNN.crossValidation(CVFold, corpus);

			} else 
				System.out.println("Classifier has not developed yet!");
		}
		else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize, multipleLearner);
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW")) {
//				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, 1, 1, 5, 1, 0, 1e-4, 1, false);
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, classNumber, featureSize, multipleLearner, sr, 100, 50, 1, 0.1, 1e-4, eta, false);
				mySemi.setFeaturesLookup(analyzer.getFeaturesLookup()); //give the look up to the classifier for debugging purpose.
				mySemi.setDebugOutput(debugOutput);
//				mySemi.setDebugPrinters(WrongRWfile, WrongSVMfile, FuSVM);
//				mySemi.setMatrixA(analyzer.loadMatrixA(matrixFile));
				mySemi.crossValidation(CVFold, corpus);
//				mySemi.printReviewStat(reviewStatFile);
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
