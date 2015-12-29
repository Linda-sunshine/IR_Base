package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import Analyzer.UserAnalyzer;
import Classifier.semisupervised.CoLinAdapt.asyncCoLinAdaptFirstOrder;
import opennlp.tools.util.InvalidFormatException;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/SelectedVocab.csv"; // CV.
		String userFolder = "./data/CoLinAdapt/Users";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.config(trainRatio, adaptRatio);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF", 0);	
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
//		//Create the instances of a LinAdapt model.
//		LinAdapt linAdapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		linAdapt.loadUsers(analyzer.getUsers());
//		
//		linAdapt.train();
//		linAdapt.test();
		
//		//Create the instances of an asyncLinAdapt model.
//		asyncLinAdapt aLinAdaptS = new asyncLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		aLinAdaptS.loadUsers(analyzer.getUsers());
//		
//		aLinAdaptS.train();
//		aLinAdaptS.test();
//		
		//Create the instances of a CoLinAdapt model.
//		CoLinAdapt coLinAdapt = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		coLinAdapt.loadUsers(analyzer.getUsers());
//		
//		coLinAdapt.train();
//		coLinAdapt.test();
//		
//		//Create the instances of an zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt coLinAdaptZeroOrder = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		coLinAdaptZeroOrder.loadUsers(analyzer.getUsers());
//		
//		coLinAdaptZeroOrder.train();
//		coLinAdaptZeroOrder.test();
//		
//		//Create the instances of an zero-order asyncCoLinAdapt model.
		asyncCoLinAdaptFirstOrder coLinAdaptFirstOrder = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		coLinAdaptFirstOrder.loadUsers(analyzer.getUsers());
		
		coLinAdaptFirstOrder.train();
		coLinAdaptFirstOrder.test();
	}
}
