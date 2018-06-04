package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.LinkPredictionWithMMB;
import Application.LinkPredictionWithMMBPerEdge;
import Application.LinkPredictionWithSVM;
import Application.LinkPredictionWithSVMPerEdge;
import Application.LinkPredictionWithSVMWithText;

public class MyLinkPredMain {
	
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
		
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;
		String fs = "DF";//"IG_CHI"
		String prefix = "./data/CoLinAdapt";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users", prefix, dataset);
		
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
//		String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
		String trainFriendFile = String.format("%s/%s/%sFriends_train.txt", prefix, dataset, dataset);
		String testFriendFile = String.format("%s/%s/%sFriends_test.txt", prefix, dataset, dataset);
		int order = 1;
		String nonFriendFile = String.format("%s/%s/%sNonFriends_order_%d.txt", prefix, dataset, dataset, order);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(trainFriendFile);
		analyzer.loadTestFriendship(testFriendFile);
		analyzer.buildNonFriendship(nonFriendFile);
		analyzer.checkFriendSize();
		
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.001, eta = 0.01, beta = 0.01;
		double sdA = 0.0425, sdB = 0.0425;
		double c = 1, rho = 0.05;
		
		String model = "mmb_edge";
		LinkPredictionWithMMB linkPred = null;

		if(model.equals("mmb_node"))
			linkPred = new LinkPredictionWithMMB();
		else if(model.equals("mmb_edge"))
			linkPred = new LinkPredictionWithMMBPerEdge(analyzer.getTrainMap(), analyzer.getTestMap());
		else if(model.equals("svm"))
			linkPred = new LinkPredictionWithSVM(c, rho);
		else if(model.equals("svm_edge"))
			linkPred = new LinkPredictionWithSVMPerEdge(c, rho, analyzer.getTrainMap(), analyzer.getTestMap());
		else if(model.equals("svm+text"))
			linkPred = new LinkPredictionWithSVMWithText(c, rho, lmTopK);
	
		
		linkPred.initMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		linkPred.getMMB().setR2TradeOffs(eta3, eta4);
		linkPred.getMMB().setsdA(sdA);
		linkPred.getMMB().setsdB(sdB);
			
		linkPred.getMMB().setR1TradeOffs(eta1, eta2);
		linkPred.getMMB().setConcentrationParams(alpha, eta, beta);

		linkPred.getMMB().setRho(0.09);
		linkPred.getMMB().setBurnIn(0);
//		linkPred.getMMB().setThinning(5);// default 3
		linkPred.getMMB().setNumberOfIterations(1);
		
		linkPred.getMMB().loadLMFeatures(analyzer.getLMFeatures());
		linkPred.getMMB().loadUsers(analyzer.getUsers());
		linkPred.getMMB().setDisplayLv(displayLv);
		
		linkPred.getMMB().train();
		linkPred.linkPrediction();
		linkPred.calculateAllNDCGMAP();
		linkPred.calculateAvgNDCGMAP();
//		linkPred.printLinkPrediction("./", model);	
	}
}
