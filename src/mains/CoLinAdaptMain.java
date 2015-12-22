package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.UserAnalyzer;
import CoLinAdapt.CoLinAdaptSchedule;

public class CoLinAdaptMain {

	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
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
		coLinAdaptS.calcluateSimilarities();
		int topK = 15;
		coLinAdaptS.constructNeighborhood(topK);
//		//If we know the number of neighbors in advance, we can init gradients before.
//		coLinAdaptS.initGradients();
		
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
