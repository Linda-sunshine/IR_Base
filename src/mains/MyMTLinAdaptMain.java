package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithSupUsrNoAdpt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithSupUsr;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncMTLinAdapt;

public class MyMTLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		// Best performance for mt-linadapt.
		double eta1 = 0.8, eta2 = 0.3;
		//eta3 = 0.1, eta4 = 0.7;
//		double eta1 = 1, eta2 = 0.5, eta3 = 1, eta4 = 0.5, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Yelp"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
			
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		adaptRatio = 0.5;
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		// Create an instances of CoLinAdapt model.
//		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
		// Create an instances of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);

		// Create an instance of MTCoLinAdapt model.
//		double lambda1 = 0.5, lambda2 = 1;
//		MTCoLinAdapt adaptation = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		asyncCoLinAdapt adaptation = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
		// Create an instance of asyncMTLinAdapt.
//		asyncMTLinAdapt adaptation = new asyncMTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null);
//		
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setR2TradeOffs(eta3, eta4);
//		adaptation.setRsTradeOffs(lambda1, lambda2);
//		adaptation.train();
//		adaptation.test();
		
		//Create the instance of MTLinAdapt.
//		double lambda1 = 0.1, lambda2 = 0.1;
//		MTLinAdapt mtlinadapt = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
//		mtlinadapt.setPersonlized(false);
//		mtlinadapt.loadUsers(analyzer.getUsers());
//		mtlinadapt.setDisplayLv(displayLv);
//		mtlinadapt.setR1TradeOffs(eta1, eta2);
//		mtlinadapt.setRsTradeOffs(lambda1, lambda2);		
//		mtlinadapt.train();
//		mtlinadapt.test();
		
//		//Create an instance of MTLinAdpatWithSupUserNoAdpt.
//		MTLinAdaptWithSupUserNoAdpt mtlinadaptsupnoadpt = new MTLinAdaptWithSupUserNoAdpt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		mtlinadaptsupnoadpt.loadUsers(analyzer.getUsers());
//		mtlinadaptsupnoadpt.setDisplayLv(displayLv);
//		mtlinadaptsupnoadpt.setR1TradeOffs(eta1, eta2);
//		double p = 0.5, q = 0.1;
//		double beta = 0.1;
//		mtlinadaptsupnoadpt.setWsWgCoefficients(p, q);
//		mtlinadaptsupnoadpt.setR14SupCoefficients(beta);
//		mtlinadaptsupnoadpt.train();
//		mtlinadaptsupnoadpt.test();
//		double eta1 = 1, eta2 = 0.5;
//		double lambda1 = 0.1, lambda2 = 0.1;
		double[] values = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		for(double lambda2: values){
			for(double lambda1: values){
		
		//Create an instance of MTLinAdapt with Super user sharing different dimensions.
		MTLinAdaptWithSupUsr mtlinadaptsup = new MTLinAdaptWithSupUsr(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
		//mtlinadaptsup.setPersonlized(false);
		mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
		mtlinadaptsup.loadUsers(analyzer.getUsers());
		mtlinadaptsup.setDisplayLv(displayLv);
		mtlinadaptsup.setR1TradeOffs(eta1, eta2);
		mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
		mtlinadaptsup.train();
		mtlinadaptsup.test();
		}}
		//mtlinadaptsup.saveModel("./data/models/mtlinadaptsup/");
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		//mtsvm.setPersonlized(false);
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
		
		// Create the instance of individual SVM.
//		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
////		indsvm.setBias(false);
//		indsvm.train();
//		indsvm.test();
		
		// Create the instance of MTCoLinAdapt.
//		MTCoLinAdapt mtcolinadapt = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		mtcolinadapt.loadUsers(analyzer.getUsers());
//		mtcolinadapt.setDisplayLv(displayLv);
//		mtcolinadapt.setR1TradeOffs(eta1, eta2);
//		mtcolinadapt.setR2TradeOffs(eta3, eta4);
//		mtcolinadapt.setRsTradeOffs(lambda1, lambda2);
//		mtcolinadapt.train();
//		mtcolinadapt.test();
		
		// Create an instances of asyncLinAdapt model.
//		asyncLinAdapt asynclinadapt = new asyncLinAdapt(classNumber,analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		asynclinadapt.loadUsers(analyzer.getUsers());
//		asynclinadapt.setDisplayLv(displayLv);
//		asynclinadapt.train();
//		asynclinadapt.test();
	}
}
