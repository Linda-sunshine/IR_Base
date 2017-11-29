package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;


public class MyMMBIsoMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;

		boolean enforceAdapt = true;
 
		String dataset = "Amazon"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;
		String fs = "DF";//"IG_CHI"
		
//		String prefix = "./data/CoLinAdapt";
		String prefix = "/zf8/lg5bt/DataSigir";

//		int testSize = 2000;
//		int trainSize = 10000 - testSize;
		
		int trainSize = 0;
		for(int testSize: new int[]{7000}){
			trainSize = 10000 - testSize;
		
		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String trainFolder = String.format("%s/%s/Users_%d", prefix, dataset, trainSize);
		String testFolder =  String.format("%s/%s/Users_%d", prefix, dataset, testSize);
		
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		adaptRatio = 1; enforceAdapt = true;
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		
		// load training users with (adaptRatio=1, testRatio=0)
		analyzer.loadUserDir(trainFolder);
		
		// load testing users with (adaptaRatio=0, testRatio=1)
		adaptRatio = 0; enforceAdapt = false;
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(testFolder);
		
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.5, eta = 0.05, beta = 0.01;
		double sdA = 0.2, sdB = 0.2;

//		String model = "dp"; // "dp"
//		String perfFile = String.format("./data/%s_%s_perf_%d.txt", dataset, model, testSize);
//	 	if(model.equals("mtsvm")){
//	 		// baseline: mt-svm
//	 		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//	 		mtsvm.loadUsers(analyzer.getUsers());
//	 		mtsvm.setBias(true);
//	 		
//	 		mtsvm.train();
//	 		mtsvm.test();
//	 		mtsvm.printUserPerformance(perfFile);
//	 	} else if(model.equals("dp")){
//	 	
//	 		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup);
//	 		adaptation.setAlpha(alpha);
//
//	 		adaptation.setsdB(sdA);//0.2
//	 		adaptation.setsdA(sdB);//0.2
//		
//	 		adaptation.setR1TradeOffs(eta1, eta2);
//	 		adaptation.setBurnIn(10);
//	 		adaptation.setNumberOfIterations(30);
//		
//	 		adaptation.loadUsers(analyzer.getUsers());
//	 		//adaptation.checkTestReviewSize();
//	 		adaptation.setDisplayLv(displayLv);
//		
//	 		adaptation.train();
//	 		adaptation.test();
//	 		adaptation.printUserPerformance(perfFile);
//		}}
		
//		MTCLinAdaptWithHDP adaptation = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
//		adaptation.setR2TradeOffs(eta3, eta4);
//		adaptation.setConcentrationParams(alpha, eta, beta);
//		adaptation.loadLMFeatures(analyzer.getLMFeatures());
		
//		CLRWithMMB mmb = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setsdA(0.2);
//		
//		MTCLRWithMMB mmb = new MTCLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setQ(0.1);
//		
//		CLinAdaptWithMMB mmb = new CLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
//		mmb.setsdB(0.1);

		MTCLinAdaptWithMMB mmb = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		mmb.setR2TradeOffs(eta3, eta4);
		
		mmb.setsdA(sdA);
		mmb.setsdB(sdB);
				
		mmb.setR1TradeOffs(eta1, eta2);
		mmb.setConcentrationParams(alpha, eta, beta);

		mmb.setRho(0.01);
		mmb.setBurnIn(10);
//		mmb.setThinning(5);// default 3
		mmb.setNumberOfIterations(30);
		
		mmb.loadLMFeatures(analyzer.getLMFeatures());
		mmb.loadUsers(analyzer.getUsers());
		mmb.checkTestReviewSize();
		mmb.setDisplayLv(displayLv);					
		
		mmb.train();
		mmb.linkPrediction("cos");
		mmb.printLinkPrediction("./", testSize);
//		
//		// Print out the current related models
//		long current = System.currentTimeMillis();
//		System.out.println(current);
//		String dir = String.format("./data/mmb/%d_%s_%d", current, dataset, testSize);
//		mmb.saveEverything(dir);
		}
	}
}

