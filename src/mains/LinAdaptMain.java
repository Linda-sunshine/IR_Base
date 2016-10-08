package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import opennlp.tools.util.InvalidFormatException;
import structures._PerformanceStat.TestMode;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		//this is for batch mode
		double trainRatio = 0, adaptRatio = 0.70;
		//this is for online mode
//		double trainRatio = 0, adaptRatio = 1.0;
		int topKNeighbors = 20;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = .01, eta2 = .01, eta3 = .01, eta4 = .01, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/SelectedVocab.csv"; // CV.
		String userFolder = "./data/CoLinAdapt/Users_4000";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups_800.txt";
		String featureGroupFileB = null; //"./data/CoLinAdapt/CrossGroups_1600.txt";
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
//		//Create an instance of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);

//		//Create an instance of asyncLinAdapt model.
//		asyncLinAdapt adaptation = new asyncLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		
		//Create an instance of CoLinAdapt model.
//		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
//		//Create an instance of zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt adaptation = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

//		//Create an instance of first-order asyncCoLinAdapt model.
//		asyncCoLinAdaptFirstOrder adaptation = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), 
//				featureMap, topKNeighbors, globalModel, featureGroupFile, neighborsHistoryWeight);

		//Create an instance of Regularized LogitReg model.
//		RegLR adaptation = new RegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		
		//Create an instance of asynchronized Regularized LogitReg model.
//		asyncRegLR adaptation = new asyncRegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		
		//Create an instance of Collaboratively Regularized LogitReg model.
//		CoRegLR adaptation = new CoRegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, topKNeighbors);
		
		//Create an instance of zero-order Collaboratively Regularized LogitReg model.
//		asyncCoRegLR adaptation = new asyncCoRegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, topKNeighbors);

		//Create an instance of first-order Collaboratively Regularized LogitReg model.
//		asyncCoRegLRFirstOrder adaptation = new asyncCoRegLRFirstOrder(classNumber, analyzer.getFeatureSize(), 
//				featureMap, globalModel, topKNeighbors, neighborsHistoryWeight);

		//Create an instance of MT-LinAdapt model.
//		MTLinAdapt adaptation = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null);

		// Create an instance of MT-LinAdapt-SupUser
//		MTLinAdaptWithSupUserNoAdapt adaptation = new MTLinAdaptWithSupUserNoAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
		//Create the instance of MT-SVM
//		MultiTaskSVM adaptation = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		adaptation.setBias(true);
		
		//Create an instance of asynchronized MT-LinAdapt model.
//		asyncMTLinAdapt adaptation = new asyncMTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null);

		/** Added by lin for calling neighborhood learning.
		//The entrance for calling the CoLinAdaptWithNeighborhoodLearning.
		int fDim = 3; // xij contains <bias, bow, svd_sim>
		String svdFile = "./data/CoLinAdapt/Amazon_SVD_Scaled.mm";
		analyzer.loadSVDFile(svdFile);
		CoLinAdaptWithNeighborhoodLearning adaptation = new CoLinAdaptWithNeighborhoodLearning(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, fDim);
		*/		

		// Create an instance of CoLinAdapt with different feature groups for different classes.
//		CoLinAdaptWithDiffFeatureGroups adaptation = new CoLinAdaptWithDiffFeatureGroups(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, featureGroupFileB);
//		adaptation.setGCoefficients(-0.5, 0.5);
//		adaptation.setR1TradeOffs(eta1/2, eta2/2);
//		adaptation.setR2TradeOffs(eta3/2, eta4/2);
		
		// Create an instance of AvgAdapt on all features
//		WeightedAvgAdapt adaptation = new WeightedAvgAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
		// Create an instance of AvgAdapt on transformation operations
//		WeightedAvgTransAdapt adaptation = new WeightedAvgTransAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		// Create an instance of Multi-task DP over logistic regression 
//		CLRWithDP adaptation = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		
		// Create an instance of Multi-task DP over Multi-task logistic regression 
//		MTCLRWithDP adaptation = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);

		// Create an instance of Multi-task DP over Multi-task logistic regression 
//		CLinAdaptWithDP adaptation = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);

		// Create an instance of Multi-task DP over MT-LinAdapt 
//		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileB);
		

		CLRWithHDP adaptation = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, analyzer.getCorpus().getBackgroundProb());
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setTestMode(TestMode.TM_batch);
		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setR2TradeOffs(eta3, eta4);
		
		adaptation.train();
		adaptation.test();
	}
}
