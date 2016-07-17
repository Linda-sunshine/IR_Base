package mains;

import java.io.IOException;
import java.text.ParseException;

import Analyzer.jsonAnalyzer;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.NaiveBayesEM;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;
import influence.PageRank;
import structures._Corpus;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int classNumber = 5; //Define the number of classes
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
	
		//"SUP", "SEMI", "FV", "ASPECT"
		String style = "SEMI";
		
		//"NB", "LR", "SVM", "PR"
		String classifier = "NB-EM"; //Which classifier to use.
		//"GF", "NB-EM"
		String model = "NB-EM";
		double C = 1.0;
//		String modelPath = "./data/Model/";
		String debugOutput = null; //"data/debug/LR.output";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

//		/*****Parameters in feature selection.*****/
		String featureSelection = "CHI"; //Feature selection method.
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.5; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
//		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablet/small";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model
		
		String pattern = String.format("%dgram_%s", Ngram, featureSelection);
		String fvFile = String.format("data/Features/fv_%s_small.txt", pattern);
		String fvStatFile = String.format("data/Features/fv_stat_%s_small.txt", pattern);
		String vctFile = String.format("data/Fvs/vct_%s_tablet_small.dat", pattern);		
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
//		/****Loading json files*****/
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
//		/****Feature selection*****/
		System.out.println("Performing feature selection, wait...");
		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
		analyzer.SaveCVStat(fvStatFile);	
		
		/****create vectors for documents*****/
//		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.setReleaseContent( !(classifier.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.setFeatureValues(featureValue, norm);
////		analyzer.setTimeFeatures(window);
//		
//		_Corpus corpus = analyzer.getCorpus();
//		
//		/********Choose different classification methods.*********/
//		//Execute different classifiers.
//		if (style.equals("SUP")) {
//			if(classifier.equals("NB")){
//				//Define a new naive bayes with the parameters.
//				System.out.println("Start naive bayes, wait...");
//				NaiveBayes myNB = new NaiveBayes(corpus);
//				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
//				
//			} else if(classifier.equals("LR")){
//				//Define a new logistics regression with the parameters.
//				System.out.println("Start logistic regression, wait...");
//				LogisticRegression myLR = new LogisticRegression(corpus, C);
//				myLR.setDebugOutput(debugOutput);
//				
//				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
//				//myLR.saveModel(modelPath + "LR.model");
//			} else if(classifier.equals("SVM")){
//				System.out.println("Start SVM, wait...");
//				SVM mySVM = new SVM(corpus, C);
//				mySVM.crossValidation(CVFold, corpus);
//				
//			} else if (classifier.equals("PR")){
//				System.out.println("Start PageRank, wait...");
//				PageRank myPR = new PageRank(corpus, C, 100, 50, 1e-6);
//				myPR.train(corpus.getCollection());
//				
//			} else System.out.println("Classifier has not developed yet!");
//		} else if (style.equals("SEMI")) {
//			if (model.equals("GF")) {
//				System.out.println("Start Gaussian Field, wait...");
//				GaussianFields mySemi = new GaussianFields(corpus, classifier, C);
//				mySemi.crossValidation(CVFold, corpus); 
//			} else if (model.equals("NB-EM")) {
//				corpus.setUnlabeled();
//				
//				System.out.println("Start Naive Bayes with EM, wait...");
//				NaiveBayesEM myNB = new NaiveBayesEM(corpus);
//				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
//			}
//		} else if (style.equals("FV")) {
//			corpus.save2File(vctFile);
//			System.out.format("Vectors saved to %s...\n", vctFile);
//		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
