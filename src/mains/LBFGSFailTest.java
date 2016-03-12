package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithSupUsr;
import opennlp.tools.util.InvalidFormatException;

public class LBFGSFailTest {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 0.5, eta2 = 1;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
		String folder = String.format("/if15/lg5bt/DataSigir/%s/", dataset);
		
		MultiThreadedUserAnalyzer analyzer;
		//Access users of different size.
		int size = 5200;
		String diffFolder;
		// We need ten sets of experiments to do the average.
		double[] ps = new double[]{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		double[] qs = new double[]{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		for(double lambda1: ps){
			for(double lambda2: qs){
			diffFolder = String.format("%sUsers_%d/Users_%d", folder, 1, size);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(diffFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
				
//			//Create the instance of MT-SVM
//			MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//			mtsvm.setPersonlized(false);
//			mtsvm.loadUsers(analyzer.getUsers());
//			mtsvm.setBias(true);
//			mtsvm.train();
//			mtsvm.test();
//			F1[i] = mtsvm.getPerf();
				
			// Create instance of MTLinAdaptWithSupUsr
//			double lambda1 = 0.1, lambda2 = 0.7;
			MTLinAdaptWithSupUsr mtlinadaptsup = new MTLinAdaptWithSupUsr(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
			mtlinadaptsup.setPersonlized(false);
			mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
			mtlinadaptsup.loadUsers(analyzer.getUsers());
			mtlinadaptsup.setDisplayLv(displayLv);
			mtlinadaptsup.setR1TradeOffs(eta1, eta2);
			mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
				
			mtlinadaptsup.train();
			mtlinadaptsup.test();
			}
		}
	}
}
