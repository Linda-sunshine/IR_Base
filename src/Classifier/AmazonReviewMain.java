package Classifier;

import java.io.IOException;
import structures._Corpus;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException{
		_Corpus corpus = new _Corpus();
		
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		String classifier = "SVM"; //Which classifier to use.
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tClassifier: " + classifier);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablets";
		String suffix = ".json";
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
		int DFthreshold = 5; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/*****Paramters in time series analysis.*****/
		int window = 7;
		boolean timeFlag = true;
		
		if( providedCV.isEmpty() && featureSelection.isEmpty()){	
			
			//Case 1: no provided CV, no feature selection.
			System.out.println("Case 1: no provided CV, no feature selection.  Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, null, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			//featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation);
			analyzer.setFeatureValues(corpus, featureValue);
		} else if( !providedCV.isEmpty() && featureSelection.isEmpty()){
			
			//Case 2: provided CV, no feature selection.
			System.out.println("Case 2: provided CV, no feature selection. Start loading files, wait...");
			jsonAnalyzer jsonAnalyzer = new jsonAnalyzer(tokenModel, classNumber, providedCV, null, Ngram, timeFlag, window);
			jsonAnalyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			//featureSize = analyzer.getFeatureSize();
//			corpus = jsonAnalyzer.returnCorpus(finalLocation); 
//			jsonAnalyzer.setFeatureValues(corpus, featureValue);
		} else if(providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 3: no provided CV, feature selection.
			System.out.println("Case 3: no provided CV, feature selection. Start loading files to do feature selection, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb, DFthreshold); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram, timeFlag, window);
			//DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);//featureLocation contains the selected features.
			analyzer_2.LoadDirectory(folder, suffix);
			//featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, featureValue);
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, featureSelection, Ngram);
			System.out.println("Case 4: provided CV, feature selection. Start loading files to do feature selection, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb, DFthreshold); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);
			analyzer_2.LoadDirectory(folder, suffix);
			//featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, featureValue);
		} else System.out.println("The setting fails, please check the parameters!!");
		
		//Execute different classifiers.
		if(classifier.equals("NB")){
			//Define a new naive bayes with the parameters.
			System.out.println("Start naive bayes, wait...");
			NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize);
			myNB.crossValidation(10, corpus, classNumber);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("LR")){
			double lambda = 0; //Define a new lambda.
			//Define a new logistics regression with the parameters.
			System.out.println("Start logistic regression, wait...");
			LogisticRegression myLR = new LogisticRegression(corpus, classNumber, featureSize, lambda);
			myLR.crossValidation(10, corpus, classNumber);//Use the movie reviews for testing the codes.
		} else if(classifier.equals("SVM")){
			//corpus.save2File("data/FVs/fvector.dat");
			double C = 3;// The default value is 1.
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(corpus, classNumber, featureSize, C);
			mySVM.crossValidation(10, corpus, classNumber);
		} else System.out.println("Have not developed yet!:(");
	}
}
