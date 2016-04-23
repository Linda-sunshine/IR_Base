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
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncMTLinAdapt;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;

public class MyMTLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		// Best performance for CoLinAdapt.
//		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859;

		// Best performance for mt-linadapt in amazon.
		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		double eta3 = 0.1, eta4 = 0.3;
//		double eta1 = 0.1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		// Best performance for mt-linadapt in yelp.
//		double eta1 = 0.9, eta2 =1 , lambda1 = 0.1, lambda2 = 0.1;
		boolean enforceAdapt = true;
		double[] perf;
		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
				
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", dataset, 800);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", dataset, 800);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

		//Create an instance of MTLinAdapt with Super user sharing different dimensions.
//		MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null); 
////		mtlinadaptsup.setLNormFlag(false);
//		mtlinadaptsup.loadUsers(analyzer.getUsers());
//		mtlinadaptsup.setDisplayLv(displayLv);
//		mtlinadaptsup.setR1TradeOffs(eta1, eta2);
//		mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
//		mtlinadaptsup.train();
//		mtlinadaptsup.test();
//		perf = mtlinadaptsup.getPerf();
//		mtlinadaptsup.saveSupModel("mtlinadapt_supModel_yelp.txt");
//		mtlinadaptsup.saveModel(String.format("./%s_mtlinadapt", dataset));
	
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		// Create the instance of global SVM.
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.setBias(false);
//		gsvm.train();
//		gsvm.test();
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.setBias(true);
//		gsvm.train();
//		gsvm.test();
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		// Create the instance of individual SVM.
//		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.setBias(false);
//		indsvm.train();
//		indsvm.test();
		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.setBias(true);
//		indsvm.train();
//		indsvm.test();
		
		// Create an intance of RegLR.
//		RegLR adaptation = new RegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		
		// Create an instances of LinAdapt model.
		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, numberOfCores, globalModel,featureGroupFile);
		
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();

//		adaptation.saveModel(String.format("./ACLModels/%s_reglr", dataset));
//		adaptation.saveModel(String.format("./ACLModels/%s_linadapt", dataset));
		
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		//mtsvm.setPersonlized(false);
//		mtsvm.loadUsers(analyzer.getUsers());
////		mtsvm.setBias(false);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.saveModel(String.format("./ACLModels/%s_mtsvm", dataset));
		
		// Create the instance of MTCoLinAdapt.
//		MTCoLinAdapt mtcolinadapt = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		mtcolinadapt.loadUsers(analyzer.getUsers());
//		mtcolinadapt.setDisplayLv(displayLv);
//		mtcolinadapt.setR1TradeOffs(eta1, eta2);
//		mtcolinadapt.setR2TradeOffs(eta3, eta4);
//		mtcolinadapt.setRsTradeOffs(lambda1, lambda2);
//		mtcolinadapt.train();
//		mtcolinadapt.test();
	}
}
