package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._User;
import utils.Utils;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;

public class MyLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		int topKNeighbors = 100;

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05, neighborsHistoryWeight = 0.5;
		double lambda1 = 0.1, lambda2 = 0.3;

		boolean enforceAdapt = true;

		String dataset = "Yelp"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

				
//		CLRWithDP adaptation = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setLNormFlag(false);
//		((CLogisticRegressionWithDP) adaptation).setR1TradeOffs(eta1, eta2);
//		adaptation.train();
//		adaptation.test();
//		((CLogisticRegressionWithDP) adaptation).printInfo();
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		// Create an instance of CLinAdaptWithDP
//		CLinAdaptWithDP adaptation = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
	
		// Create an instance of MTCLRWithDP
//		MTCLRWithDP adaptation = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);	
		
		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setNumberOfIterations(15);
		adaptation.setAlpha(1);
		adaptation.setsdA(0.05);
//		adaptation.setQ(q);
		adaptation.setsdB(0.05);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();
		adaptation.printInfo();
		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();

//		eta1 = 1; eta2 = 0.5; lambda1 = 0.1; lambda2 = 0.3;
//		// Create an instances of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.train();
//		adaptation.test();
//		 
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		eta1 = 1; eta2 = 0.5; lambda1 = 0.1; lambda2 = 0.3;
//		MTLinAdapt adaptation  = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null); 
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setRsTradeOffs(lambda1, lambda2);
//		adaptation.train();
//		adaptation.test();	
	}
}
