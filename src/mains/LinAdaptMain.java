package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.UserAnalyzer;
import CoLinAdapt.LinAdaptSchedule;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int featureGroupNo = 400; //There should be some feature grouping methods.
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/LinAdapt/selectedFeatures.csv"; // CV.
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		
		//Load users too.
		String folder = "./data/LinAdapt/Users";
		String featureGroupFile = "./data/LinAdapt/CrossGroups.txt";
		String globalModel = "./data/LinAdapt/global.classifer";
		analyzer.loadUserDir(folder);
		analyzer.setFeatureValues("BM25", 2);
		analyzer.loadFeatureGroupIndexes(featureGroupFile);
		
		double[] globalWeights = analyzer.loadGlobalWeights(globalModel);
		//Create the instances of the LinAdapt model.
		LinAdaptSchedule linAdaptS = new LinAdaptSchedule(analyzer.getUsers(), analyzer.getFeatureSize(), featureGroupNo, analyzer.getFeatureGroupIndexes());
		linAdaptS.setGlobalWeights(globalWeights);
		linAdaptS.init();
		
//		//Online training.
//		linAdapt.onlineTrain();
//		linAdapt.calcOnlinePerformance(); //Calculate the performance of each user.	
		
		//Batch training.
		linAdaptS.batchTrainTest();
		linAdaptS.calcBatchPerformance();
	}
}
