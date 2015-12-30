package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import Analyzer.UserAnalyzer;
import Classifier.semisupervised.CoLinAdapt.CoLinAdapt;
import opennlp.tools.util.InvalidFormatException;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 10;
		int displayLv = 1;
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/SelectedVocab.csv"; // CV.
		String userFolder = "./data/CoLinAdapt/Users";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.config(trainRatio, adaptRatio);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
//		//Create the instances of a LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);

//		//Create the instances of an asyncLinAdapt model.
//		asyncLinAdapt adaptation = new asyncLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		
		//Create the instances of a CoLinAdapt model.
		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
//		//Create the instances of an zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt adaptation = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

//		//Create the instances of an zero-order asyncCoLinAdapt model.
//		asyncCoLinAdaptFirstOrder adaptation = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		
		adaptation.train();
		adaptation.test();
	}
}
