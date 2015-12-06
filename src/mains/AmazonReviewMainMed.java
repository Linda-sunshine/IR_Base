package mains;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;

import structures._Corpus;
import Analyzer.appReviewAnalyzer;
import Analyzer.jsonAnalyzer;
import Analyzer.medforumAnalyzer;
import Analyzer.newEggAnalyzer;
import Classifier.semisupervised.GaussianFields;
import Classifier.supervised.EMNaiveBayes;
import Classifier.supervised.EMNaiveBayes;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.NaiveBayesEM;
import Classifier.supervised.SVM;

public class AmazonReviewMainMed {

	public static void main(String[] args) throws IOException, ParseException{
		/*****Set these parameters before run the classifiers.*****/
		int classNumber = 2; //Define the number of classes
		int Ngram = 1; //The default value is bigram. 
		
		//"TF", "TFIDF", "BM25", "PLN"
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 2;//The way of normalization.(only 1 and 2)
		int CVFold = 5; //k fold-cross validation
	
		//"SUP", "SEMI", "FV", "ASPECT"
		String style = "SUP";
		
		//"NB", "LR", "SVM", "PR", "EM-NB"
		String classifier = "LR";//Which classifier to use.
		double C = 1.0;
		String debugOutput = null; //"data/debug/LR.output";
		
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tLearning Method: " + style + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

//		/*****Parameters in feature selection.*****/
		String featureSelection = "DF"; //Feature selection method.
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.3; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
//		System.out.println("Feature Selection: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****The parameters used in loading files.*****/

		//String[] products = {"camera","tablet", "laptop", "phone", "surveillance", "tv"};
		// change topic number and category
		String category = "Games"; //Utilities, Travel, Sports, Social_Networking, Reference, Productivity, Photo_and_Video, News, Navigation, Music, Food_and_Drink, Finance, Education, Medical, Games, Business, Health_and_Fitness, Lifestyle
		
		boolean m_LoadnewEggInTrain = false;
		int lengthThreshold = 2; //Document is now sentence length threshold
		boolean m_randomFold = true;
		int numberOfIteration = 200; // for EM-Naive Bayes
		double varConverge = 1e-5; // for EM-Naive Bayes
		
		String resultPath = "./model/"+classifier+"/result/med/"+category+"_information.txt";
		String folder1 = "/home/nahid/workspace/medforumCrawler/data/json/eHealth/Depression/";
		String folder2 = "/home/nahid/workspace/medforumCrawler/data/json/eHealth/Eating_Disorder/";
		String unfolder = "./data/app/newdata/un-annotated/";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		
		String pattern = String.format("%dgram_%s_%s", Ngram, featureValue, featureSelection);
		String featureDirectory = "./model/"+classifier+"/result/";
		
		String fvFile = String.format("%sfv_%s_small.txt", featureDirectory, pattern);
		String fvStatFile = String.format("%sfv_stat_%s_small.txt", featureDirectory, pattern);
		String vctFile = String.format("%svct_%s_small.dat", featureDirectory, pattern);		
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		System.out.println("Performing feature selection, wait...");
		medforumAnalyzer analyzer = new medforumAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setCategory("Depression");
		analyzer.setClassifierOrTopicModel(true,stnModel);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder1, suffix);
		analyzer.setCategory("Eating-Disorder");
		analyzer.LoadDirectory(folder2, suffix);
		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		analyzer.SaveCVStat(fvStatFile);	
		
		
		
		/****create vectors for documents*****/
		System.out.println("Creating feature vectors, wait...");
		analyzer = new medforumAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.setClassifierOrTopicModel(true,stnModel);
		analyzer.setReleaseContent( !(classifier.equals("PR") || debugOutput!=null) );//Just for debugging purpose: all the other classifiers do not need content
		
		analyzer.setCategory("Depression");
		analyzer.LoadDirectory(folder1, suffix);
		analyzer.setCategory("Eating-Disorder");
		analyzer.LoadDirectory(folder2, suffix);
		
		analyzer.setFeatureValues(featureValue, norm);
		
		_Corpus corpus = analyzer.getCorpus();
		ArrayList<String> featureSet = corpus.getAllFeatures();
		int topK = featureSet.size();
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (style.equals("SUP")) {
			if(classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus);
				myNB.crossValidation(CVFold, corpus);
				myNB.printTopFeaturesSet(topK, featureSet);
				
			}
			else if(classifier.equals("EM-NB")){
				//Define a new EM-naive bayes with the parameters.
				System.out.println("Start EM-naive bayes, wait...");
				EMNaiveBayes myNB = new EMNaiveBayes(corpus,numberOfIteration,varConverge);
				myNB.setInfoWriter(resultPath);
				myNB.crossValidation(CVFold, corpus);
			}
			else if(classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, C);
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
				
			} else System.out.println("Classifier has not developed yet!");
		} else if (style.equals("SEMI")) {
			GaussianFields mySemi = new GaussianFields(corpus, classifier, C);
			mySemi.crossValidation(CVFold, corpus);
		} else if (style.equals("FV")) {
			corpus.save2File(vctFile);
			System.out.format("Vectors saved to %s...\n", vctFile);
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
