package Classifier;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TFIDF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		String classifier = "LR"; //Which classifier to use.
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tClassifier: " + classifier);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablets";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
//		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/data/movie/FinalFeatureStat.txt"; //The destination of storing the final features with stats.
//		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/data/movie/SelectedFeatures.txt";
		String finalLocation = "./FinalFeatureStat.txt";
		String featureLocation = "./SelectedFeatures.txt";

		/*****Parameters in feature selection.*****/
		String providedCV = "";
		String featureSelection = "IG"; //Feature selection method.
		double startProb = 0.4; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int DFthreshold = 5; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Pre-process the data.*****/
		//Feture selection.
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer jsonAnalyzer = new jsonAnalyzer(tokenModel, classNumber, providedCV, featureSelection, Ngram);
//		jsonAnalyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		jsonAnalyzer.featureSelection(featureLocation, startProb, endProb, DFthreshold); //Select the features.
		
		//Collect vectors for documents.
		featureSelection = "";
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer jsonAnalyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, featureSelection, Ngram);
		jsonAnalyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		jsonAnalyzer.setFeatureValues(featureValue, norm);
		jsonAnalyzer.setTimeFeatures(window);
		featureSize = jsonAnalyzer.getFeatureSize();
		_Corpus corpus = jsonAnalyzer.returnCorpus(finalLocation);
		
		corpus.save2File("./fv.dat");
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if(classifier.equals("NB")){
			//Define a new naive bayes with the parameters.
			System.out.println("Start naive bayes, wait...");
			NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize + window);
			myNB.crossValidation(10, corpus);//Use the movie reviews for testing the codes.
			
		} else if(classifier.equals("LR")){
			//Define a new logistics regression with the parameters.
			double lambda = 0; //Define a new lambda.
			System.out.println("Start logistic regression, wait...");
			LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize + window, lambda);
			myLR.crossValidation(10, corpus);//Use the movie reviews for testing the codes.
			
		} else if(classifier.equals("SVM")){
			//corpus.save2File("data/FVs/fvector.dat");
			double C = 3;// The default value is 1.
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(corpus, classNumber, featureSize + window, C);
			mySVM.crossValidation(10, corpus);
			
		} else System.out.println("Have not developed yet!:(");
	}
}
