package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import Analyzer.UserAnalyzer;
import CoLinAdapt.LinAdaptSchedule;

import opennlp.tools.util.InvalidFormatException;


public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int featureGroupNo = 400; //There should be some feature grouping methods.
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/5000Features.txt"; // CV.
		String CVStat = "./data/CoLinAdapt/SelectedVocab.csv";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";

		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		String folder = "./data/CoLinAdapt/Users2";

		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold); //CV is loaded here.
		analyzer.LoadCVStat(CVStat);
//		analyzer.LoadStopwords("./data/Model/custom.stop"); //No need since stop words are not in features.

		analyzer.setReleaseContent(false);//Do not release the content of the review.
		analyzer.loadUserDir(folder); //Load users.		
		analyzer.setFeatureValues(25265, "TFIDF", 0);
	
		analyzer.loadGlobalWeights(globalModel);
		analyzer.loadFeatureGroupIndexes(featureGroupFile);
//		linAdaptS.setFeatures(analyzer.getFeatures());

		//Create the instances of the LinAdapt model.
		LinAdaptSchedule linAdaptS = new LinAdaptSchedule(analyzer.getUsers(), analyzer.getFeatureSize(), featureGroupNo, analyzer.getFeatureGroupIndexes());
		linAdaptS.setGlobalWeights(analyzer.getGlobalWeights());
		linAdaptS.initSchedule();

		//Online training.
		linAdaptS.onlineTrain();
		linAdaptS.calcPerformance(); //Calculate the performance of each user.	
		linAdaptS.printPerformance();
		
		//Batch training.
		linAdaptS.initSchedule();
		linAdaptS.batchTrainTest();
		linAdaptS.calcPerformance();
		linAdaptS.printPerformance();

	}
}
