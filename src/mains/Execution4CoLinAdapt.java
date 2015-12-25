package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import structures.CoLinAdaptParameter;
import Analyzer.UserAnalyzer;
import CoLinAdapt.CoLinAdaptSchedule;

public class Execution4CoLinAdapt {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		CoLinAdaptParameter param = new CoLinAdaptParameter(args);
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int featureGroupNo = 400; //There should be some feature grouping methods.
	
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/5000Features.txt"; // CV.// CV.
		String CVStat = "./data/CoLinAdapt/SelectedVocab.csv";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		String folder = "./data/CoLinAdapt/Amazon/Users";

		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.LoadCVStat(CVStat);

		analyzer.loadUserDir(folder);//Load users.	
		analyzer.setFeatureValues(25265, "TFIDF", 0);
	
		analyzer.loadGlobalWeights(globalModel);
		analyzer.loadFeatureGroupIndexes(featureGroupFile);
	
		//Create the instances of the LinAdapt model.
		CoLinAdaptSchedule coLinAdaptS = new CoLinAdaptSchedule(analyzer.getUsers(), analyzer.getFeatureSize(), featureGroupNo, analyzer.getFeatureGroupIndexes());
		coLinAdaptS.setGlobalWeights(analyzer.getGlobalWeights());
		coLinAdaptS.initSchedule();
//		int topK = 15;
		String neighborFile = "./data/CoLinAdapt/Neighbor";
		coLinAdaptS.calcluateSimilarities();
		coLinAdaptS.constructNeighborhood(param.m_topK);
	
//		coLinAdaptS.writeUserNeighbors(neighborFile, topK);
//		coLinAdaptS.loadUserNeighbors(neighborFile, param.m_topK);
	
		//Online training.
		coLinAdaptS.onlineTrain();
		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
		coLinAdaptS.printPerformance();
	
		//Batch training.
		coLinAdaptS.initSchedule();
		coLinAdaptS.batchTrainTest();
		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
		coLinAdaptS.printPerformance();
	}
}
