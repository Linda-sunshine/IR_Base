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
		String providedCV = "./data/CoLinAdapt/selectedFeatures.csv"; // CV.
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		//Load users.
		String folder = "./data/CoLinAdapt/Amazon/Users";
//		String folder = "./data/CoLinAdapt/Amazon/Users";

		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/global.classifer";
		analyzer.loadUserDir(folder);
		analyzer.setFeatureValues("BM25", 2);
		analyzer.loadFeatureGroupIndexes(featureGroupFile);
		
		double[] globalWeights = analyzer.loadGlobalWeights(globalModel);
		//Create the instances of the LinAdapt model.
		CoLinAdaptSchedule coLinAdaptS = new CoLinAdaptSchedule(analyzer.getUsers(), analyzer.getFeatureSize(), featureGroupNo, analyzer.getFeatureGroupIndexes());
		coLinAdaptS.setGlobalWeights(globalWeights);
		coLinAdaptS.initSchedule();
		coLinAdaptS.calcluateSimilarities();
		int topK = 20;
		coLinAdaptS.constructNeighborhood(topK);
//		//If we know the number of neighbors in advance, we can init gradients before.
//		coLinAdaptS.initGradients();
		
		//Online training.
		coLinAdaptS.onlineTrain();
		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
		coLinAdaptS.printPerformance();
		
		//Batch training.
//		coLinAdaptS.batchTrainTest();
//		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
//		coLinAdaptS.printPerformance();
	}
}
