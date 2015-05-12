package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.AspectAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.semisupervised.GaussianFields;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.SVM;

public class AmazonReviewMain {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "BM25"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
	
		//"SUP", "SEMI", "FV", "ASPECT"
		String style = "ASPECT";
		
		//"NB", "LR", "SVM", "PR"
		String classifier = "LR"; //Which classifier to use.
		double C = 0.1;
//		String modelPath = "./data/Model/";
		String debugOutput = "data/debug/LR.output";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

//		/*****Parameters in feature selection.*****/
		String featureSelection = "DF"; //Feature selection method.
		int chiSize = 50; // top ChiSquare words for aspect keyword selection
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1.0; // Used in feature selection, the ending point of the features.
		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
//		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Token model.
		
		String aspectModel = "./data/Model/aspectlist.txt"; // list of keywords in each aspect
		String aspectOutput = "./data/Model/aspect_output.txt"; // list of keywords in each aspect
		String pattern = String.format("%dgram_%s_%s", Ngram, featureValue, featureSelection);
		String fvFile = String.format("data/Features/fv_%s.txt", pattern);
		String fvStatFile = String.format("data/Features/fv_stat_%s.txt", pattern);
		String vctFile = String.format("data/Fvs/vct_%s.dat", pattern);		
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		/****Loading json files*****/
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold,
				aspectModel, chiSize);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
		/****Feature selection*****/
//		System.out.println("Performing feature selection, wait...");
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
//		analyzer.SaveCVStat(fvStatFile);
		
		/****Aspect annotation*****/
		analyzer.BootStrapping(aspectOutput, 0.9, 10);		
		
		/****create vectors for documents*****/
//		System.out.println("Creating feature vectors, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, finalLocation, Ngram, lengthThreshold);
//		analyzer.setReleaseContent( !(classifier.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.setFeatureValues(featureValue, norm);
//		analyzer.setTimeFeatures(window);
		
		featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.getCorpus();
		
		//temporal code to add pagerank weights
//		PageRank tmpPR = new PageRank(corpus, classNumber, featureSize + window, C, 100, 50, 1e-6);
//		tmpPR.train(corpus.getCollection());
		
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
			GaussianFields mySemi = new GaussianFields(corpus, classNumber, featureSize + window + 1, classifier);
			mySemi.crossValidation(CVFold, corpus);
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
