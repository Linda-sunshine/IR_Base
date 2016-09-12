package mains;

import java.io.IOException;

import Analyzer.DocAnalyzer;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;
import structures._Corpus;

public class MovieReviewMain {
	/*****************************Main function*******************************/
	public static void main(String[] args) throws IOException{
		_Corpus corpus = new _Corpus();
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 1;
		String classifier = "SVM"; //Which classifier to use.
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tClassifier: " + classifier);

		/*****The parameters used in loading files.*****/
		String folder = "data/txt_sentoken";
		String suffix = ".txt";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
//		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/data/movie/FinalFeatureStat.txt"; //The destination of storing the final features with stats.
//		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/data/movie/SelectedFeatures.txt";
		String finalLocation = "/home/lin/Lin'sWorkSpace/IR_Base/FinalFeatureStat.txt";
		String featureLocation = "/home/lin/Lin'sWorkSpace/IR_Base/SelectedFeatures.txt";

		/*****Paramters in feature selection.*****/
		//String providedCV = "";
		String featureSelection = "";
		String providedCV = "Features.txt"; //Provided CV.
		//String featureSelection = "MI"; //Feature selection method.
		double startProb = 0.5; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int maxDF = -1, minDF = 5; // Filter the features with DFs smaller than this threshold.
		
		
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		System.out.println("--------------------------------------------------------------------------------------");
//		/*****Paramters in time series analysis.*****/
//		int window = 7;
//		boolean timeFlag = true;
		
		if( providedCV.isEmpty() && featureSelection.isEmpty()){	
			
			//Case 1: no provided CV, no feature selection.
			System.out.println("Case 1: no provided CV, no feature selection.  Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.setFeatureValues(featureValue, norm);
			corpus = analyzer.returnCorpus(finalLocation); 
		} else if( !providedCV.isEmpty() && featureSelection.isEmpty()){
			
			//Case 2: provided CV, no feature selection.
			System.out.println("Case 2: provided CV, no feature selection. Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.setFeatureValues(featureValue, norm);
			corpus = analyzer.returnCorpus(finalLocation); 
		} else if(providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 3: no provided CV, feature selection.
			System.out.println("Case 3: no provided CV, feature selection. Start loading files to do feature selection, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
			
			System.out.println("Start loading files, wait...");
			analyzer = new DocAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
			analyzer.LoadDirectory(folder, suffix);
			analyzer.setFeatureValues(featureValue, norm);
			corpus = analyzer.returnCorpus(finalLocation); 
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
			System.out.println("Case 4: provided CV, feature selection. Start loading files to do feature selection, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
			
			System.out.println("Start loading files, wait...");
			analyzer = new DocAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
			analyzer.LoadDirectory(folder, suffix);
			analyzer.setFeatureValues(featureValue, norm);
			corpus = analyzer.returnCorpus(finalLocation); 
		} else System.out.println("The setting fails, please check the parameters!!");
		
		//Execute different classifiers.
		if(classifier.equals("NB")){
			//Define a new naive bayes with the parameters.
			System.out.println("Start naive bayes, wait...");
			NaiveBayes myNB = new NaiveBayes(corpus);
			myNB.crossValidation(10, corpus);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("LR")){
			double lambda = 0; //Define a new lambda.
			//Define a new logistics regression with the parameters.
			System.out.println("Start logistic regression, wait...");
			LogisticRegression myLR = new LogisticRegression(corpus, lambda);
			myLR.crossValidation(10, corpus);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("SVM")){
			//corpus.save2File("data/FVs/fvector.dat");
			double C = 3;// The default value is 1.
			double eps = 0.01;// default value from Lin's implementation
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(corpus, C);
			mySVM.crossValidation(10, corpus);
		} else System.out.println("Have not developed yet!:(");
	}
}
