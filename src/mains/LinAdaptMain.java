package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import Analyzer.UserAnalyzer;
import Classifier.semisupervised.CoLinAdapt.LinAdapt;
import opennlp.tools.util.InvalidFormatException;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/selectedFeatures.csv"; // CV.
		String userFolder = "./data/CoLinAdapt/Users";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/global.classifer";
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("BM25", 2);		
		
		//Create the instances of the LinAdapt model.
		LinAdapt linAdaptS = new LinAdapt(classNumber, analyzer.getFeatureSize(), globalModel, featureGroupFile);
		linAdaptS.loadUsers(analyzer.getUsers());
	}
}
