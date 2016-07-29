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

//		/***baseline 1: global***/
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		String dir = String.format("./data/wsdm/%s_", dataset);
//		/***baseline 2: mtsvm***/
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.saveModel(dir + "mtsvm_0.5/");
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
//		clindp.setsdA(0.2);
//		clindp.setsdB(0.1);
//		clindp.train();
//		clindp.test();
//		clindp.printInfo();		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		/***baseline 7: CLinAdaptWithKmeans***/
		// We perform kmeans over user weights learned from individual svms.
		int kmeans = 25;
		int[] clusters;
		KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
		alg.train(analyzer.getUsers());
		clusters = alg.getClusters();// The returned clusters contain the corresponding cluster index of each user.
		
		CLinAdaptWithKmeans clinkmeans = new CLinAdaptWithKmeans(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, kmeans, clusters);
		clinkmeans.loadUsers(analyzer.getUsers());
		clinkmeans.setDisplayLv(displayLv);
		clinkmeans.setLNormFlag(false);
		clinkmeans.setR1TradeOffs(eta1, eta2);
		clinkmeans.train();
		clinkmeans.test();
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
//		/***our algorithm: MTCLinAdaptWithDP***/
//		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setLNormFlag(false);
//		adaptation.setsdA(0.2);
//		adaptation.setsdB(0.1);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setR2TradeOffs(eta3, eta4);
//		adaptation.train();
//		adaptation.test();
//		adaptation.printInfo();
////		adaptation.saveClusterModel(dir + "mtclindp_c_0.5/");
////		adaptation.saveModel(dir + "mtclindp_u_0.5");
////		adaptation.saveClusterInfo(dir + "clusterInfo.txt");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
	}
}
