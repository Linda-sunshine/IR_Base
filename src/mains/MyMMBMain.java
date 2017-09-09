package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.MMB.CLRWithMMB;
import Classifier.supervised.modelAdaptation.MMB.CLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MMB.MTCLRWithMMB;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMBDocFirst;


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
		
		String friendFile = String.format("%s/%s/Friends.txt", prefix, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		double[] globalLM = analyzer.estimateGlobalLM();
		
//		CLRWithMMB mmb = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setsdA(0.2);
		
//		MTCLRWithMMB mmb = new MTCLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setQ(0.1);
		
//		CLinAdaptWithMMB mmb = new CLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
//		mmb.setsdB(0.1);//0.2

		MTCLinAdaptWithMMB mmb = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		mmb.setR2TradeOffs(eta3, eta4);
//
//		MTCLinAdaptWithMMBDocFirst mmb = new MTCLinAdaptWithMMBDocFirst(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
//		mmb.setR2TradeOffs(eta3, eta4);
		
		mmb.setsdA(0.1);//0.2	
		double alpha = 1, eta = 0.1, beta = 0.01;
		mmb.setConcentrationParams(alpha, eta, beta);
		mmb.setR1TradeOffs(eta1, eta2);

		mmb.setRho(0.01);
		mmb.setBurnIn(10);
//		mmb.setThinning(5);// default 3
		mmb.setNumberOfIterations(50);
		mmb.loadLMFeatures(analyzer.getLMFeatures());
		mmb.loadUsers(analyzer.getUsers());
		mmb.setDisplayLv(displayLv);					
		
		mmb.train();
		mmb.test();
		mmb.printStat();
		
	}
}
