package mains;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.VctAnalyzer;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.KNN;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.PRLogisticRegression;
import Classifier.supervised.SVM;

public class VectorReviewMain {

	public static void main(String[] args) throws IOException, ParseException{		
		/*****Set these parameters before run the classifiers.*****/
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int lengthThreshold = 5; //Document length threshold
		
		int CVFold = 10; //k fold-cross validation
		
		//Supervised classification models: "NB", "LR", "PR-LR", "SVM"
		//Semi-supervised classification models: "GF", "GF-RW", "GF-RW-ML"
		String classifier = "GF-RW-ML"; //Which classifier to use.
//		String modelPath = "./data/Model/";
		double C = 1.0;
		
		//"SUP", "SEMI"
		String style = "SEMI";
		String multipleLearner = "SVM";
		
		/*****The parameters used in loading files.*****/
		String featureLocation = "data/Features/fv_2gram_BM25_CHI_small.txt";
		String vctfile = "data/FVs/vct_2gram_BM25_CHI_tablet_small.dat";
		
//		String featureLocation = "data/Features/fv_fake.txt";
//		String vctfile = "data/Fvs/LinearRegression.dat";
		
		/*****Parameters in time series analysis.*****/
		//String debugOutput = String.format("data/debug/%s.sim.pair", classifier);
		String debugOutput = null;
		
		/****Pre-process the data.*****/
		//Feture selection.
		System.out.println("Loading vectors from file, wait...");
		VctAnalyzer analyzer = new VctAnalyzer(classNumber, lengthThreshold, featureLocation);
		analyzer.LoadDoc(vctfile); //Load all the documents as the data set.
				
		_Corpus corpus = analyzer.getCorpus();
		
		/********Choose different classification methods.*********/
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus);
				myNB.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(classifier.equals("KNN")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start kNN, wait...");
				KNN myKNN = new KNN(corpus, 10, 1);
				myKNN.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				
			}  else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, C);
				myLR.setDebugOutput(debugOutput);
				
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("PRLR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start posterior regularized logistic regression, wait...");
				PRLogisticRegression myLR = new PRLogisticRegression(corpus, C);
				myLR.setDebugOutput(debugOutput);
				
				myLR.crossValidation(CVFold, corpus);//Use the movie reviews for testing the codes.
				//myLR.saveModel(modelPath + "LR.model");
			} else if(classifier.equals("SVM")){
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, C);
				mySVM.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("PR")){
				System.out.println("Start PageRank, wait...");
				PageRank myPR = new PageRank(corpus, C, 100, 50, 1e-6);
				myPR.train(corpus.getCollection());
				
			} else System.out.println("Classifier has not been developed yet!");
		} else if (style.equals("SEMI")) {
			if (classifier.equals("GF")) {
				GaussianFields mySemi = new GaussianFields(corpus, multipleLearner, C);
				mySemi.crossValidation(CVFold, corpus);
				
			} else if (classifier.equals("GF-RW")) {
				GaussianFields mySemi = new GaussianFieldsByRandomWalk(corpus, multipleLearner, C,
						0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false);
				mySemi.setDebugOutput(debugOutput);
				
				mySemi.crossValidation(CVFold, corpus);
			} else if (classifier.equals("GF-RW-ML")) {
				LinearSVMMetricLearning lMetricLearner = new LinearSVMMetricLearning(corpus, multipleLearner, C,
						0.1, 100, 50, 1.0, 0.1, 1e-4, 0.1, false,
						2);
				lMetricLearner.setMetricLearningMethod(true);
				//lMetricLearner.setDebugOutput(debugOutput);
				
				lMetricLearner.crossValidation(CVFold, corpus);
			} else System.out.println("Classifier has not been developed yet!");
			
		} else System.out.println("Learning paradigm has not been developed yet!");
	} 

}
