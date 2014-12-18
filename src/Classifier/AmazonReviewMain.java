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
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 5; //k fold-cross validation
		
		//"NB", "LR", "SVM"
		String classifier = "NB"; //Which classifier to use.
		
		//"SUP", "TRANS"
		String style = "SUP";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearing Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stopwords = "./data/Model/stopwords.dat";
		String finalLocation = "./data/Features/FinalFeatureStat.txt";
		String featureLocation = "./data/Features/SelectedFeatures.txt";

		/*****Parameters in feature selection.*****/
		String featureSelection = "CHI"; //Feature selection method.
		double startProb = 0.4; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Pre-process the data.*****/
//		//Feture selection.
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		//Collect vectors for documents.
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		analyzer.setTimeFeatures(window);
		featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.returnCorpus(finalLocation);
		
		//corpus.save2File("./data/FVs/fv.dat");
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize + window);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				double lambda = 1.0; //Define a new lambda.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize + window, lambda);
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("SVM")){
				//corpus.save2File("data/FVs/fvector.dat");
				double C = 3;// The default value is 1.
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, classNumber, featureSize + window, C);
				mySVM.crossValidation(CVFold, corpus);
				
			} else System.out.println("Classifier has not developed yet!");
		} else if (style.equals("TRANS")) {
			SemiSupervised mySemi = new SemiSupervised(corpus, classNumber, featureSize + window, classifier);
			mySemi.crossValidation(CVFold, corpus);
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
