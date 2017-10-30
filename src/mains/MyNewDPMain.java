package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;

public class MyNewDPMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.75;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

//		double eta1 = 0.005, eta2 = 0.005, eta3 = 0.0025, eta4 = 0.0025;
		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;

		int trainSize = 3; // "3"
		int userSize = 20; // "20"
		String dataset = "AmazonNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
//		int[] kFolds = new int[]{5, 10};
//		int[] kmeanss = new int[]{200, 400, 600, 800, 1000};
	
		int kFold = 10, kmeans = 400;
		int lrTopK = 3000, lmTopK = 1000; // topK for language model.
		String fs1 = "IG", fs2 = "CHI";
		
//		String fvFile = String.format("./data/CoLinAdapt/Amazon/SelectedVocab.csv"); // CV.
//		String fvFile4LM = String.format("./data/CoLinAdapt/Amazon/fv_lm.txt");
//		String globalModel = String.format("./data/CoLinAdapt/Amazon/GlobalWeights.txt");
//		String crossfv = String.format("./data/CoLinAdapt/Amazon/CrossGroups_800.txt");

		String userDir = String.format("./data/%s/Users_%dk", dataset, userSize);
		String fvFile = String.format("./data/%s/fv_%dk_%s_%s_%d.txt", dataset, trainSize, fs1, fs2, lrTopK);
		String fvFile4LM = String.format("./data/%s/fv_%dk_lm_%d_DF.txt", dataset, trainSize, lmTopK);
		String globalModel = String.format("./data/%s/GlobalWeights_%dk_%d.txt", dataset, trainSize, lrTopK);
		String crossfv = String.format("./data/%s/CrossFeatures_%dk_%d_%d_%d/", dataset, trainSize, lrTopK, kFold, kmeans);

//		String trainDir = String.format("/if15/lg5bt/%s/Users_%dk", dataset, trainSize);
//		String userDir = String.format("/if15/lg5bt/%s/Users_%dk", dataset, userSize);
//		String fvFile = String.format("/if15/lg5bt/%s/fv_%dk_%s_%s_%d.txt", dataset, trainSize, fs1, fs2, lrTopK);
//		String fvFile4LM = String.format("/if15/lg5bt/%s/fv_%dk_lm_%d_DF.txt", dataset, trainSize, lmTopK);
//		String globalModel = String.format("/if15/lg5bt/%s/GlobalWeights_%dk.txt", dataset, trainSize);
//		String crossfv = String.format("/if15/lg5bt/%s/CrossFeatures_%dk_%d_%d/", dataset, trainSize, kFold, kmeans);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, fvFile, fvFile4LM, Ngram, lengthThreshold, numberOfCores, true);
		String ctgFile = "./data/category_AmazonNew.txt";
//		analyzer.setCtgFile(ctgFile);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userDir);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		
		double[] globalLM = analyzer.estimateGlobalLM();
//		//analyzer.saveCategory(ctgFile);
//		analyzer.printCategoryInfo();
		
//		double[] globalLM = analyzer.estimateGlobalLM();
//		CLRWithHDP hdp = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, globalLM);
		
//		MTCLRWithHDP hdp = new MTCLRWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, globalLM);
//		hdp.setQ(0.6);
		
//		MTCLinAdaptWithDP dp = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), globalModel, crossfv, null);
//
//		MTCLinAdaptWithDPExp dp = new MTCLinAdaptWithDPExp(classNumber, analyzer.getFeatureSize(), globalModel, crossfv, null);
//		dp.loadUsers(analyzer.getUsers());
//		dp.setLNormFlag(false);
//		dp.setDisplayLv(displayLv);
//		dp.setNumberOfIterations(30);
//		dp.setsdA(0.1);
//		dp.setsdB(0.1);
//		dp.setR1TradeOffs(eta1, eta2);
//		dp.setR2TradeOffs(eta3, eta4);
//		dp.train();
//		dp.test();
		
//		MTCLinAdaptWithHDPExp hdp = new MTCLinAdaptWithHDPExp(classNumber, analyzer.getFeatureSize(), globalModel, crossfv, null, globalLM);

//		MTCLinAdaptWithHDP hdp = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), globalModel, crossfv, null, globalLM);
//		hdp.setR2TradeOffs(eta3, eta4);
//		hdp.setsdB(0.1);
//
//		hdp.setsdA(0.1);
//		double alpha = 1, eta = 0.01, beta = 0.01;
//		hdp.setConcentrationParams(alpha, eta, beta);
//		hdp.setR1TradeOffs(eta1, eta2);
//		hdp.setNumberOfIterations(20);
//		hdp.loadUsers(analyzer.getUsers());
//		hdp.setDisplayLv(displayLv);
//		hdp.train();
//		hdp.test();
		
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
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
