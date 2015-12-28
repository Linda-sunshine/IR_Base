package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.UserAnalyzer;
import Classifier.semisupervised.CoLinAdapt.CoLinAdapt;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 5;
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/5000Features.txt"; // CV.
		String userFolder = "./data/CoLinAdapt/Users";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.config(trainRatio, adaptRatio);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("BM25", 2);		
		
		//Create the instances of a LinAdapt model.
//		LinAdapt linAdapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), globalModel, featureGroupFile);
//		linAdapt.loadUsers(analyzer.getUsers());
//		
//		linAdapt.train();
//		linAdapt.test();
		
		//Create the instances of an asyncLinAdapt model.
//		asyncLinAdapt aLinAdaptS = new asyncLinAdapt(classNumber, analyzer.getFeatureSize(), globalModel, featureGroupFile);
//		aLinAdaptS.loadUsers(analyzer.getUsers());
//		
//		aLinAdaptS.train();
//		aLinAdaptS.test();
		
		//Create the instances of a CoLinAdapt model.
		CoLinAdapt coLinAdapt = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), topKNeighbors, globalModel, featureGroupFile);
		coLinAdapt.loadUsers(analyzer.getUsers());
		
		coLinAdapt.train();
		coLinAdapt.test();
		
		//Create the instances of an zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt coLinAdaptZeroOrder = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), topKNeighbors, globalModel, featureGroupFile);
//		coLinAdaptZeroOrder.loadUsers(analyzer.getUsers());
//		
//		coLinAdaptZeroOrder.train();
//		coLinAdaptZeroOrder.test();
		
		//Create the instances of an zero-order asyncCoLinAdapt model.
//		asyncCoLinAdaptFirstOrder coLinAdaptZeroOrder = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), topKNeighbors, globalModel, featureGroupFile);
//		coLinAdaptZeroOrder.loadUsers(analyzer.getUsers());
//		
//		coLinAdaptZeroOrder.train();
//		coLinAdaptZeroOrder.test();
	}
}
