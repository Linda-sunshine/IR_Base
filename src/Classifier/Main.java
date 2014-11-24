package Classifier;

import java.io.IOException;

import structures._Corpus;
import Analyzer.DocAnalyzer;

public class Main {

	/*****************************Main function*******************************/
	public static void main(String[] args) throws IOException{
		
		//Set these parameters before run the classifiers.
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be tfidf, BM25
		String classifier = "SVM"; //Which classifier to use.
		
		System.out.println("*******************************************************************************************************************");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tClassifier: " + classifier);
		
		_Corpus corpus = new _Corpus();

		//The parameters used in loading files.
		String folder = "txt_sentoken";
		String suffix = ".txt";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/FinalFeatureStat.txt"; //The destination of storing the final features with stats.
		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/SelectedFeatures.txt";

//		String finalLocation = "/home/lin/Lin'sWorkSpace/IR_Base/FinalFeatureStat.txt";
//		String featureLocation = "/home/lin/Lin'sWorkSpace/IR_Base/SelectedFeatures.txt";

		String providedCV = "";
		//String featureSelection = "";
		//String providedCV = "Features.txt"; //Provided CV.
		
		String featureSelection = "MI"; //Feature selection method.
		double startProb = 0.8; // Used in feature selection, the starting point of the features.
		double endProb = 0.95; // Used in feature selection, the ending point of the feature.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		System.out.println("*******************************************************************************************************************");

		if( providedCV.isEmpty() && featureSelection.isEmpty()){	
			
			//Case 1: no provided CV, no feature selection.
			System.out.println("Case 1: no provided CV, no feature selection.");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, null, Ngram);
			System.out.println("Start loading files, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation);
			analyzer.setFeatureValues(corpus, featureValue);
		} else if( !providedCV.isEmpty() && featureSelection.isEmpty()){
			
			//Case 2: provided CV, no feature selection.
			System.out.println("Case 2: provided CV, no feature selection.");
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, null, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation); 
			analyzer.setFeatureValues(corpus, featureValue);
		} else if(providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 3: no provided CV, feature selection.
			System.out.println("Case 3: no provided CV, feature selection.");
			System.out.println("Start loading files to do feature selection, wait...");

			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);//featureLocation contains the selected features.
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, featureValue);
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, featureSelection, Ngram);
			System.out.println("Case 4: provided CV, feature selection.");
			System.out.println("Start loading file to do feature selection, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, featureValue);
		} else{
			System.out.println("The setting fails. please check the parameters!!");
		}
		
		//Execute different classifiers.
		if(classifier.equals("NB")){
			//Define a new naive bayes with the parameters.
			System.out.println("Start naive bayes, wait...");
			NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize);
			System.out.println("Start cross validaiton, wait...");
			myNB.crossValidation(10, corpus, classNumber);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("LR")){
			double lambda = 0; //Define a new lambda.
			//Define a new logistics regression with the parameters.
			System.out.println("Start logistic regression, wait...");
			LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize, lambda);
			System.out.println("Start cross validaiton, wait...");
			myLR.crossValidation(10, corpus, classNumber);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("SVM")){
			corpus.save2File("data/FVs/fvector.dat");
			double C = 3;// The default value is 1.
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(corpus, classNumber, featureSize, C);
			mySVM.crossValidation(5, corpus, classNumber);
		} else{
			System.out.println("Have not developed yet!:(");
		}
	}
}
