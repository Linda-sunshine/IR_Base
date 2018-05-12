package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.LinkPrediction;

public class MyLinkPredBoWMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;
 
		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;
		String fs = "DF";//"IG_CHI"
		String prefix = "./data/CoLinAdapt";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		for(int train: new int[]{6000}){
		String userPrefix = "/home/lin/DataSigir";
		String trainUserFolder = String.format("%s/%s/Users_%d_train", userPrefix, dataset, train);
		String testUserFolder = String.format("%s/%s/Users_2000_test", userPrefix, dataset);

		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(trainUserFolder);
		
		adaptRatio = 0;
		analyzer.config(trainRatio, adaptRatio, false);
		analyzer.loadUserDir(testUserFolder);
		
		analyzer.buildFriendship(friendFile);
		analyzer.checkFriendSize();
		
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.001, eta = 0.01, beta = 0.01;
		double sdA = 0.0425, sdB = 0.0425;
		
		LinkPrediction linkPred = new LinkPrediction();
		
		linkPred.initMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
//		linkPred.getMMB().setR2TradeOffs(eta3, eta4);
//		linkPred.getMMB().setsdA(sdA);
//		linkPred.getMMB().setsdB(sdB);
//			
//		linkPred.getMMB().setR1TradeOffs(eta1, eta2);
//		linkPred.getMMB().setConcentrationParams(alpha, eta, beta);
//
//		linkPred.getMMB().setRho(0.09);
//		linkPred.getMMB().setBurnIn(0);
////		linkPred.getMMB().setThinning(5);// default 3
//		linkPred.getMMB().setNumberOfIterations(1);
//		
//		linkPred.getMMB().loadLMFeatures(analyzer.getLMFeatures());
		linkPred.getMMB().loadUsers(analyzer.getUsers());
		linkPred.getMMB().setDisplayLv(displayLv);
		
//		linkPred.getMMB().train();
		linkPred.linkPrediction();
		linkPred.calculateAllNDCGMAP();
		linkPred.calculateAvgNDCGMAP();}
//		linkPred.printLinkPrediction("./", model);	
	}
}
