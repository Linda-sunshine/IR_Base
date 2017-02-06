package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.CLRWithMMB;


public class MyMMBMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
//		double eta1 = 0.1, eta2 = 0.1, eta3 = 0.04, eta4 = 0.04;

		boolean enforceAdapt = true;

		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 5000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;

		String fs = "DF";//"IG_CHI"
		
		String prefix = "./data/CoLinAdapt";
//		String prefix = "/if15/lg5bt/DataSigir";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users_1000", prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = "./data/CoLinAdapt/YelpNew/yelpFriends_1000.txt";
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		/*****hdp related models.****/
		double[] globalLM = analyzer.estimateGlobalLM();
		CLRWithMMB hdp = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		hdp.setR2TradeOffs(eta3, eta4);
//		hdp.setsdB(0.2);

		hdp.setsdA(0.2);
		double alpha = 1, eta = 0.1, beta = 0.01;
		hdp.setConcentrationParams(alpha, eta, beta);
		hdp.setR1TradeOffs(eta1, eta2);
		hdp.setNumberOfIterations(30);
		hdp.loadUsers(analyzer.getUsers());
		hdp.setDisplayLv(displayLv);

		hdp.train();
		hdp.test();
//		
//		String perfFile = String.format("./data/hdp_lm_%d_10k.xls", lmTopK);
//		hdp.printUserPerformance(perfFile);
	}
}
