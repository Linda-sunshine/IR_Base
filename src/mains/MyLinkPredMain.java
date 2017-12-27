package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.LinkPredictionWithMMB;
import Application.LinkPredictionWithSVM;
import Application.LinkPredictionWithSVMSplit;

public class MyLinkPredMain {
	
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

		int trainSize = 200, testSize = 800;
		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String trainFolder = String.format("%s/%s/Users_%d", prefix, dataset, trainSize);
		String testFolder =  String.format("%s/%s/Users_%d", prefix, dataset, testSize);
		
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends_1000.txt", prefix, dataset, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		adaptRatio = 1; enforceAdapt = true;
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		
		// load training users with (adaptRatio=1, testRatio=0)
		analyzer.loadUserDir(trainFolder);
		
		// load testing users with (adaptaRatio=0, testRatio=1)
		adaptRatio = 0; enforceAdapt = false;
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(testFolder);
		
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.01, eta = 0.01, beta = 0.01;
		double sdA = 0.0425, sdB = 0.0425;
		double c = 1;
		
		String model = "svm_pred"; // "svm", "svm_prep", "svm_pred"
		LinkPredictionWithMMB linkPred = null;

		String trainFile = String.format("./data/trainFile_%s_%d.txt", model, trainSize);
		String testFile = String.format("./data/testFile_%s_%d.txt", model, testSize);
		//link_pred_svm_alpha_0.005
		if(model.equals("svm_pred")){
			trainFile = String.format("./data/trainFile_svm_prep_%d.txt", trainSize);
			testFile = String.format("./data/testFile_svm_prep_%d.txt", testSize);
		}
		if(model.equals("svm_pred")){
			linkPred = new LinkPredictionWithSVMSplit(c);
			((LinkPredictionWithSVMSplit) linkPred).loadData(trainFile, testFile, friendFile);
		} else if(model.equals("mmb"))
			linkPred = new LinkPredictionWithMMB();
		else if(model.equals("svm"))
			linkPred = new LinkPredictionWithSVM(c);
		else if(model.equals("svm_prep"))
			linkPred = new LinkPredictionWithSVMSplit(c);
		
		if(!model.equals("svm_pred")){
			linkPred.initMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
			linkPred.getMMB().setR2TradeOffs(eta3, eta4);
			linkPred.getMMB().setsdA(sdA);
			linkPred.getMMB().setsdB(sdB);
				
			linkPred.getMMB().setR1TradeOffs(eta1, eta2);
			linkPred.getMMB().setConcentrationParams(alpha, eta, beta);

			linkPred.getMMB().setRho(0.1);
			linkPred.getMMB().setBurnIn(10);
//			linkPred.getMMB().setThinning(5);// default 3
			linkPred.getMMB().setNumberOfIterations(5);
		
			linkPred.getMMB().loadLMFeatures(analyzer.getLMFeatures());
			linkPred.getMMB().loadUsers(analyzer.getUsers());
			linkPred.getMMB().calculateFrdStat();
			linkPred.getMMB().checkTestReviewSize();
			linkPred.getMMB().setDisplayLv(displayLv);
		
			linkPred.getMMB().train();
		}
		
		if(!model.equals("svm_prep")){
			boolean linkPredMultiThread = false;
			if(linkPredMultiThread)
				linkPred.linkPrediction_MultiThread();
			else
				linkPred.linkPrediction();
			linkPred.printLinkPrediction("./", model, trainSize, testSize);	
		} else {
			((LinkPredictionWithSVMSplit) linkPred).linkPrediction_Prep(trainFile, testFile);
		}
	}
}
