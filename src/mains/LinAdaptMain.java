package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.DocAnalyzer;
import Analyzer.UserAnalyzer;
import CoLinAdapt.LinAdapt;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/LinAdapt/selectedFeatures.csv"; // CV.
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		
		//Load users too.
		String folder = "./data/LinAdapt/Users";
		String featureGroupFile = "./data/LinAdapt/CrossGroups.txt";
		String globalModel = "./data/LinAdapt/global.classifer";
		analyzer.loadUserDir(folder);
		analyzer.fillFeatureGroups(featureGroupFile);
		
		double[] globalWeights = analyzer.loadGlobalWeights(globalModel);
		//Create the instance of the LinAdapt model.
		LinAdapt linAdapt = new LinAdapt(analyzer.getUsers(), analyzer.getFeatureSize(), analyzer.getFeatureGroupNo(), analyzer.getFeatureGroupIndex());
		linAdapt.setGlobalWeights(globalWeights);
		linAdapt.init();
		linAdapt.onlineTrain();
		linAdapt.calcPerformance(); //Calculate the performance of each user.	
	}
}
