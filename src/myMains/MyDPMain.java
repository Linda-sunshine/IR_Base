package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;

public class MyDPMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
	
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;
	
		String dataset = "Amazon"; // "Amazon", "YelpNew"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		 
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;

		String fs = "DF";//"IG_CHI"
		
		String prefix = "./data/CoLinAdapt";
//		String prefix = "/if15/lg5bt/DataSigir";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users", prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
//		for(int i=1; i<6; i++){
//		adaptRatio = i * 0.1;
		
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		//Amazon parameters.
//		double eta1 = 0.06, eta2 = 0.01, eta3 = 0.06, eta4 = 0.01;
		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;

		//Yelp parameters.
//		double eta1 = 0.09, eta2 = 0.02, eta3 = 0.07, eta4 = 0.03;
//		
//		/***baseline 0: base***/
//		Base base = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		base.loadUsers(analyzer.getUsers());
//		base.setPersonalizedModel();
//		base.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 1: global***/
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
//		//gsvm.printUserPerformance(String.format("./data/gsvm_perf_0.%d.txt", i));
//		
//		//gsvm.saveSupModel("./data/gsvm_weights.txt");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//	
//		/***baseline 2: individual svm***/
//		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.train();
//		indsvm.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 3: LinAdapt***/
//		//Create an instances of LinAdapt model.
//		LinAdapt linadapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		linadapt.loadUsers(analyzer.getUsers());
//		linadapt.setDisplayLv(displayLv);
//		linadapt.setR1TradeOffs(eta1, eta1);
//		linadapt.train();
//		linadapt.test();
//		linadapt.saveModel(String.format("./data/%s_linadapt_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//		u.getPerfStat().clear();
//
//		/***baseline 4: CLinAdaptWithKmeans***/
//		// We perform kmeans over user weights learned from individual svms.
//		int kmeans = 200;
//		int[] clusters;
//		KMeansAlg4Vct alg = new KMeansAlg4Vct(analyzer.getUsers(), kmeans, analyzer.getFeatureSize());
//		alg.train();
//		clusters = alg.getClusters();// The returned clusters contain the corresponding cluster index of each user.
//		
//		CLinAdaptWithKmeans clinkmeans = new CLinAdaptWithKmeans(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, kmeans, clusters);
//		clinkmeans.loadUsers(analyzer.getUsers());
//		clinkmeans.setDisplayLv(displayLv);
//		clinkmeans.setLNormFlag(false);
//		clinkmeans.setR1TradeOffs(eta1, eta1);
//		clinkmeans.train();
//		clinkmeans.test();
//		clinkmeans.setParameters(0, 1, 1);
////		clinkmeans.saveModel(String.format("./data/%s_clinkmeans_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
//		/***baseline 5: CLinAdaptWithDP***/
//		// Create an instance of CLinAdaptWithDP
//		CLinAdaptWithDP clindp = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		clindp.loadUsers(analyzer.getUsers());
//		clindp.setDisplayLv(displayLv);
//		clindp.setLNormFlag(false);
//		clindp.setR1TradeOffs(eta1, eta1);
//		clindp.setsdA(sdA);
//		clindp.setsdB(sdB);
//		clindp.train();
//		clindp.test();
////		clindp.saveModel(String.format("./data/%s_clindp_0.5_1", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 6: CLRWithDP***/
//		CLRWithDP clrdp = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		clrdp.loadUsers(analyzer.getUsers());
//		clrdp.setDisplayLv(displayLv);
//		clrdp.setLNormFlag(false);
//		clrdp.setR1TradeOffs(eta1, eta1);
//		clrdp.train();
//		clrdp.test();
//		clrdp.saveModel(String.format("./data/%s_clrdp_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 7: mtsvm***/
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.printUserPerformance("./data/mtsvm_perf_train.txt");
//		mtsvm.printGlobalUserPerformance("./data/mtsvm_global_perf_train.txt");
//		mtsvm.saveModel(String.format("./data/mtsvm_train/", dataset));
//		mtsvm.saveSupModel("./data/mtsvm_global_weights_train.txt");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
//		/***baseline 8: MTCLRWithDP***/
//		// Create an instance of MTCLRWithDP
//		MTCLRWithDP mtclrdp = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);	
//		mtclrdp.loadUsers(analyzer.getUsers());
//		mtclrdp.setDisplayLv(displayLv);
//		mtclrdp.setLNormFlag(false);
//		mtclrdp.setQ(0.4);
//		mtclrdp.setsdA(sdA);
//		mtclrdp.setR1TradeOffs(eta1, eta2);
//		mtclrdp.train();
//		mtclrdp.test();
//		mtclrdp.savePerf("./data/mtclrdp_perf.txt");
//		mtclrdp.saveModel(String.format("./data/%s_mtclrdp_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
		double sdA = 0.3, sdB = 0.3;
		/***our algorithm: MTCLinAdaptWithDP***/
		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup);

		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setNumberOfIterations(30);
		adaptation.setsdA(sdA);
		adaptation.setsdB(sdB);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);
		
		adaptation.train();
		adaptation.test();
		adaptation.saveModel("./data/Amazon_mtclindp_0.5_1");
		
//		adaptation.printUserPerformance("dp_exp_10k.xls");

//		MultiTaskLR mtlr = new MultiTaskLR(classNumber, analyzer.getFeatureSize());
//		mtlr.setLambda(0.1);
//		mtlr.loadUsers(analyzer.getUsers());
//		mtlr.train();
//		mtlr.test();
//		mtlr.saveModel("./data/mtlr_models");
		
//		String testPerfFile = String.format("./data/mtsvm_perf_test_adapt_%.1f.txt", adaptRatio);
//		String testPerfGlobalFile = String.format("./data/mtsvm_global_perf_test_adapt_%.1f.txt", adaptRatio);
//		mtsvm.printUserPerformance(testPerfFile);
//		mtsvm.printGlobalUserPerformance(testPerfGlobalFile);
		
//		String trainPerfFile = String.format("./data/0422mtsvm_perf_train_adapt_2_all.txt");
//		String trainPerfGlobalFile = String.format("./data/0422mtsvm_global_perf_train_adapt_2_all.txt", adaptRatio);
//		mtsvm.printUserPerformance(trainPerfFile);
//		mtsvm.printGlobalUserPerformance(trainPerfGlobalFile);
//		}
	}
}
