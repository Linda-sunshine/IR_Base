package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import cern.jet.random.Beta;
import opennlp.tools.util.InvalidFormatException;
import structures._Review;
import structures._User;
import structures._Review.rType;
import topicmodels.LDA.LDA_Gibbs;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.ReTrain;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDPExp;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.CLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.IndSVMWithKmeans;
import Classifier.supervised.modelAdaptation.HDP.IndSVMWithKmeansExp;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPConfidence;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPDualConfidence;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPExp;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPLR;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPMultipleE;

public class MyHDPMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		String stopwords = "./data/Model/stopwords.dat";

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
//		double eta1 = 0.1, eta2 = 0.1, eta3 = 0.04, eta4 = 0.04;

		boolean enforceAdapt = true;

		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		//int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;
		String fs = "DF";//"IG_CHI"
		int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.

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
		
		/**** Feature selection for language model.***/
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.loadUserDir(userFolder);
//		analyzer.featureSelection(lmFvFile, "DF", maxDF, minDF, lmTopK);
//		analyzer.featureSelection(lmFvFile, "IG", "CHI", maxDF, minDF, lmTopK);
//		int number_of_topics = 40, topK = 30;
//		String topWordPath = String.format("./data/topWords_%d_topics_top%d.txt", number_of_topics, topK);
//		
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
//		
//		/***Analyzer used for the sanity check of splitting the users.***/
//		adaptRatio = 1; int k = 400;
//		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, null, Ngram, lengthThreshold, numberOfCores, false);
//		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
//		analyzer.loadUserDir(userFolder);
//		analyzer.separateUsers(k);
//		analyzer.setFeatureValues("TFIDF-sublinear", 0);
//		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
//		
//		MTCLinAdaptWithDP dp = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
//
//		MTCLinAdaptWithDPExp dp = new MTCLinAdaptWithDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
//		dp.loadUsers(analyzer.getUsers());
//		dp.setLNormFlag(false);
//		dp.setDisplayLv(displayLv);
//		dp.setNumberOfIterations(30);
//		dp.setsdA(0.2);
//		dp.setsdB(0.2);
//		dp.setR1TradeOffs(eta1, eta2);
//		dp.setR2TradeOffs(eta3, eta4);
//		dp.train();
//		dp.test();
//		dp.printUserPerformance("./data/dp_exp_full_10k.xls");
//		
//		/*****hdp related models.****/
		double[] globalLM = analyzer.estimateGlobalLM();
//		
//		CLRWithHDP hdp = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//
//		MTCLRWithHDP hdp = new MTCLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		hdp.setQ(1);
//		
//		CLinAdaptWithHDP hdp = new CLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
//				
//		MTCLinAdaptWithHDPLR hdp = new MTCLinAdaptWithHDPLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
//
//		MTCLinAdaptWithHDP hdp = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		MTCLinAdaptWithHDPMultipleE hdp = new MTCLinAdaptWithHDPMultipleE(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		hdp.loadLMFeatures(analyzer.getLMFeatures());
		hdp.setR2TradeOffs(eta3, eta4);
		hdp.setsdB(0.1);//0.2

		hdp.setsdA(0.1);//0.2
		double alpha = 1, eta = 0.1, beta = 0.01;
		hdp.setConcentrationParams(alpha, eta, beta);
		hdp.setR1TradeOffs(eta1, eta2);
		
		hdp.setBurnIn(20);
//		hdp.setThinning(5);// default 3
//		hdp.setNumberOfIterations(100);// default 50
		hdp.loadUsers(analyzer.getUsers());
		hdp.setDisplayLv(displayLv);
		
		hdp.train();
		hdp.test();
//		hdp.printUserPerformance("./data/hdp_perf_train.txt");
//		hdp.printGlobalUserPerformance("./data/hdp_global_perf_train.txt");
	}
}
