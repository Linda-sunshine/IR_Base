package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import clustering.KMeansAlg4Profile;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._User;
import utils.Utils;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithKmeans;
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

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		String dir = "./data/";

//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users_1000", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String dir = String.format("/if15/lg5bt/DataWsdm2017/%s/%s_", dataset, dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

//		/***baseline 1: global***/
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
//		gsvm.printUserPerf(dir+"gsvm.txt");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//	
//		/***baseline 2: mtsvm***/
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.printUserPerf(dir+"mtsvm.txt");
//		//mtsvm.saveModel(dir + "mtsvm_0.5/");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 3: CLRWithDP***/
//		CLRWithDP clrdp = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		clrdp.loadUsers(analyzer.getUsers());
//		clrdp.setDisplayLv(displayLv);
//		clrdp.setLNormFlag(false);
//		clrdp.setR1TradeOffs(eta1, eta2);
//		clrdp.train();
//		clrdp.test();
//		clrdp.printInfo();
//		clrdp.printUserPerf(dir+"clrdp.txt");
//		//clrdp.saveModel(dir+"clrdp_0.5/");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 4: MTCLRWithDP***/
//		// Create an instance of MTCLRWithDP
//		MTCLRWithDP mtclrdp = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);	
//		mtclrdp.loadUsers(analyzer.getUsers());
//		mtclrdp.setDisplayLv(displayLv);
//		mtclrdp.setLNormFlag(false);
//		mtclrdp.setQ(0.4);
//		mtclrdp.setR1TradeOffs(eta1, eta2);
//		mtclrdp.train();
//		mtclrdp.test();
//		mtclrdp.printInfo();
//		mtclrdp.printUserPerf(dir+"mtclrdp.txt");
//		//mtclrdp.saveModel(dir+"mtclrdp_0.5/");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 5: LinAdapt***/
//		// Create an instances of LinAdapt model.
//		eta1 = 1; eta2 = 0.5;
//		LinAdapt linadapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		linadapt.loadUsers(analyzer.getUsers());
//		linadapt.setDisplayLv(displayLv);
//		linadapt.setR1TradeOffs(eta1, eta2);
//		linadapt.train();
//		linadapt.test();
//		linadapt.printUserPerf(dir+"linadapt.txt");
//		//linadapt.saveModel(dir+"linadapt_0.5/");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 6: CLinAdaptWithDP***/
//		// Create an instance of CLinAdaptWithDP
//		eta1 = 0.05; eta2 = 0.05; eta3 = 0.05; eta4 = 0.05;
//		CLinAdaptWithDP clindp = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		clindp.loadUsers(analyzer.getUsers());
//		clindp.setDisplayLv(displayLv);
//		clindp.setLNormFlag(false);
//		clindp.setR1TradeOffs(eta1, eta2);
//		clindp.setsdA(0.05);
//		clindp.setsdB(0.01);
//		clindp.train();
//		clindp.test();
//		clindp.printInfo();	
//		clindp.printUserPerf(dir+"clindp.txt");
//		//clindp.saveModel(dir+"clindp_0.5");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 7: CLinAdaptWithKmeans***/
//		// We perform kmeans over user weights learned from individual svms.
//		int kmeans = 25;
//		int[] clusters;
//		KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
//		alg.train(analyzer.getUsers());
//		clusters = alg.getClusters();// The returned clusters contain the corresponding cluster index of each user.
//		
//		CLinAdaptWithKmeans clinkmeans = new CLinAdaptWithKmeans(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, kmeans, clusters);
//		clinkmeans.loadUsers(analyzer.getUsers());
//		clinkmeans.setDisplayLv(displayLv);
//		clinkmeans.setLNormFlag(false);
//		clinkmeans.setR1TradeOffs(eta1, eta2);
//		clinkmeans.train();
//		clinkmeans.test();
//		clinkmeans.printUserPerf(dir+"clinkmeans.txt");
//		//clinkmeans.saveModel(dir+"clinkmeans_0.5");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		int[] group = new int[]{400, 800, 1600, 5000};
//		for(int i=0;i<group.length;i++){
//			for(int j=i; j<group.length; j++){
//		
		//Yelp best parameter: 0.23 0.1 0.04 0.01
//		double sdA = 0.23, sdB = 0.1; eta1 = 0.04; eta3 = 0.01; 
		double sdA = 0.4, sdB = 0.2; eta1 = 0.005; eta3 = 0.005; 
//		featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_%d.txt", group[i], dataset);
//		featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_%d.txt", group[j], dataset);
//		if(group[i] == 5000)
//			featureGroupFile = null;
//		if(group[j] == 5000)
//			featureGroupFileB = null;
		for(int i=0; i< 200; i++){
		/***our algorithm: MTCLinAdaptWithDP***/
		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setsdA(sdA);
		adaptation.setsdB(sdB);
		adaptation.setAlpha(1);
		adaptation.setR1TradeOffs(eta1, eta1);
		adaptation.setR2TradeOffs(eta3, eta3);
//		String traceFile = dataset + "_iter.csv";
//		adaptation.trainTrace(traceFile);
		adaptation.train();
		adaptation.test();
		adaptation.printInfo();
		adaptation.printUserPerf(String.format("%s/%s_mtclindp_%d.txt", dir, dataset, i));
//		adaptation.saveClusterModel(dir + "mtclindp_c_0.5/");
//		adaptation.saveModel(dir + "mtclindp_u_0.5");
//		adaptation.saveClusterInfo(dir + "clusterInfo.txt");
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
//		}}
		}
	}
}
