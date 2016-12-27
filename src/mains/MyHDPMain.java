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
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.CLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.IndSVMWithKmeans;
import Classifier.supervised.modelAdaptation.HDP.IndSVMWithKmeansExp;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPExp;

public class MyHDPMain {
	
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

		String dataset = "Amazon"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
		int lmTopK = 1000; // topK for language model.
		String fs = "DF";//"IG_CHI"
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users_1000", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
//		String lmFvFile = String.format("./data/CoLinAdapt/%s/fv_lm_%s_%d.txt", dataset, fs, lmTopK);
//
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
		String lmFvFile = String.format("/if15/lg5bt/DataSigir/%s/fv_lm_%d.txt", dataset, lmTopK);
		
		/**** Feature selection for language model.***/
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, false);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.loadUserDir(userFolder);
//		//analyzer.featureSelection(lmFvFile, "DF", maxDF, minDF, lmTopK);
//		analyzer.featureSelection(lmFvFile, "IG", "CHI", maxDF, minDF, lmTopK);
//		int number_of_topics = 40, topK = 30;
//		String topWordPath = String.format("./data/topWords_%d_topics_top%d.txt", number_of_topics, topK);
		adaptRatio = 1; int k = 2000;
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.separateUsers(k);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		/**Parameters in topic modeling.***/
//		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 200;//these two parameters must be larger than 1!!!
//		double converge = 1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
//		double varConverge = 1e-5, burnIn = 0.4;
//		int varIter = 10, gibbs_iteration = 1000, gibbs_lag = 50;
//		int number_of_topics = 40, topK = 30;
//		String infoFilePath = "./data/info.txt";
//		String topWordPath = String.format("./data/topWords_%d_topics_top%d.txt", number_of_topics, topK);
//		
//		LDA_Gibbs model = new LDA_Gibbs(gibbs_iteration, 0, beta, analyzer.getCorpus(), //in gibbs sampling, no need to compute log-likelihood during sampling
//			lambda, number_of_topics, alpha, burnIn, gibbs_lag);
//		model.setInforWriter(infoFilePath);
//		model.EMonCorpus();
//		model.printOnlyTopWords(topK, topWordPath);

//		IndSVMWithKmeansExp svmkmeans = new IndSVMWithKmeansExp(classNumber, analyzer.getFeatureSize(), 100);
//		svmkmeans.loadUsers(analyzer.getUsers());
//		svmkmeans.setLabel(false);
//		svmkmeans.train();
//		svmkmeans.test();
//		
//		int threshold = 100;
//		svmkmeans.CrossValidation(5, threshold);
		
//		Base base = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		base.loadUsers(analyzer.getUsers());
//		base.setPersonalizedModel();
//		base.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		MTCLinAdaptWithDPExp dp = new MTCLinAdaptWithDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
		dp.loadUsers(analyzer.getUsers());
		dp.setLNormFlag(false);
		dp.setDisplayLv(displayLv);
		dp.setNumberOfIterations(30);
		dp.setsdA(0.2);
		dp.setsdB(0.2);
		dp.setR1TradeOffs(eta1, eta2);
		dp.setR2TradeOffs(eta3, eta4);
		dp.train();
		dp.test();
		
		double[] globalLM = analyzer.estimateGlobalLM();
		
//		CLRWithHDP hdp = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		
//		MTCLRWithHDP hdp = new MTCLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		hdp.setQ(q);
//		
//		CLinAdaptWithHDP hdp = new CLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
//		MTCLinAdaptWithHDP hdp = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, null, null, globalLM);

		MTCLinAdaptWithHDPExp hdp = new MTCLinAdaptWithHDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null, globalLM);
		
		hdp.setR2TradeOffs(eta3, eta4);
		hdp.setsdB(0.1);

		hdp.setsdA(0.1);
		double alpha = 1, eta = 0.1, beta = 0.01;
		hdp.setConcentrationParams(alpha, eta, beta);
		hdp.setR1TradeOffs(eta1, eta2);
		hdp.setNumberOfIterations(30);
		hdp.loadUsers(analyzer.getUsers());
		hdp.setDisplayLv(displayLv);

		hdp.train();
		hdp.test();
		
//		int threshold = 100;
//		hdp.CrossValidation(5, threshold);
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//			
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.printEachUserPerf();
	}
}
