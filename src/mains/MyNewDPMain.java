package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.CLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;

public class MyNewDPMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;

		int trainSize = 2; // "3"
		int userSize = 20; // "20"
		String dataset = "AmazonNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.

		int kFold = 10, kmeans = 800;
		String featureSelection = "DF"; //Feature selection method.
		String pattern = String.format("%dgram_%s", Ngram, featureSelection);
		
		String trainDir = String.format("./data/%s/Users_%dk", dataset, trainSize);
		String userDir = String.format("./data/%s/Users_%dk", dataset, userSize);
		String fvFile = String.format("./data/%s/fv_%dk_%s.txt", dataset, trainSize, pattern);
//		String fvFile4LM = String.format("./data/AmazonNew/fv_%dk_lm_%d_%s.txt", trainSize, topK, pattern);
		String globalModel = String.format("./data/%s/GlobalWeights_%dk.txt", dataset, trainSize);
		String crossfv = String.format("./data/%s/CrossFeatures_%dk_%d_%d/", dataset, trainSize, kFold, kmeans);

//		String trainDir = String.format("/if15/lg5bt/%s/Users_%dk", dataset, trainSize);
//		String userDir = String.format("/if15/lg5bt/%s/Users_%dk", dataset, userSize);
//		String fvFile = String.format("/if15/lg5bt/%s/fv_%dk_%s.txt", dataset, trainSize, pattern);
////		String fvFile4LM = String.format("./data/AmazonNew/fv_%dk_lm_%d_%s.txt", trainSize, topK, pattern);
//		String globalModel = String.format("/if15/lg5bt/%s/GlobalWeights_%dk.txt", dataset, trainSize);
//		String crossfv = String.format("/if15/lg5bt/%s/CrossFeatures_%dk_%d_%d/", dataset, trainSize, kFold, kmeans);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, fvFile, null, Ngram, lengthThreshold, numberOfCores, true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userDir);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		double[] globalLM = analyzer.estimateGlobalLM();

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
		
//		double[] globalLM = analyzer.estimateGlobalLM();
//		CLRWithHDP hdp = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, globalLM);
		
//		MTCLRWithHDP hdp = new MTCLRWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, globalLM);
//		hdp.setQ(0.2);
		
		CLinAdaptWithHDP hdp = new CLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, null, globalLM);

//		MTCLinAdaptWithHDP hdp = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, crossfv, null, globalLM);
//		hdp.setR2TradeOffs(eta3, eta4);
//		hdp.setsdB(0.1);

		hdp.setsdA(0.1);
		double alpha = 1, eta = 0.1, beta = 0.1;
		hdp.setConcentrationParams(alpha, eta, beta);
		hdp.setR1TradeOffs(eta1, eta2);
		hdp.setNumberOfIterations(20);
		hdp.loadUsers(analyzer.getUsers());
		hdp.setDisplayLv(displayLv);
		hdp.train();
		hdp.test();
		
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
		mtsvm.loadUsers(analyzer.getUsers());
		mtsvm.setBias(true);
		mtsvm.train();
		mtsvm.test();
		mtsvm.printEachUserPerf();
	}
}
