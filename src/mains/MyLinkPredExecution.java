package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.DPParameter;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.LinkPredictionWithMMB;
import Application.LinkPredictionWithMMBPerEdge;
import Application.LinkPredictionWithSVM;
import Application.LinkPredictionWithSVMPerEdge;
import Application.LinkPredictionWithSVMWithText;

public class MyLinkPredExecution {
	
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		DPParameter param = new DPParameter(args);

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		boolean enforceAdapt = true;
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("%s/%s/SelectedVocab.csv", param.m_prefix, param.m_data); // CV.
		String userFolder = String.format("%s/%s/Users", param.m_prefix, param.m_data);
		
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fv);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fvSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", param.m_prefix, param.m_data);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", param.m_prefix, param.m_data, param.m_fs, param.m_lmTopK);
				
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		String trainFriendFile = String.format("%s/%s/%sFriends_train.txt", param.m_prefix,param.m_data,param.m_data);
		String testFriendFile = String.format("%s/%s/%sFriends_test.txt", param.m_prefix,param.m_data,param.m_data);
		String nonFriendFile = String.format("%s/%s/%sNonFriends_%d.txt", param.m_prefix, param.m_data, param.m_data, param.m_time);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, param.m_adaptRatio, true);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(trainFriendFile);
		analyzer.loadTestFriendship(testFriendFile);
		analyzer.buildNonFriendship(nonFriendFile);
		//analyzer.checkFriendSize();
		
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		double[] globalLM = analyzer.estimateGlobalLM();
		
		LinkPredictionWithMMB linkPred = null;
		
		if(param.m_model.equals("mmb_node"))
			linkPred = new LinkPredictionWithMMB();
		else if(param.m_model.equals("mmb_edge"))
			linkPred = new LinkPredictionWithMMBPerEdge(analyzer.getTrainMap(), analyzer.getTestMap());
		else if(param.m_model.equals("svm"))
			linkPred = new LinkPredictionWithSVM(param.m_c, param.m_rho);
		else if(param.m_model.equals("svm_edge"))
			linkPred = new LinkPredictionWithSVMPerEdge(param.m_c, param.m_rho, analyzer.getTrainMap(), analyzer.getTestMap());
		else if(param.m_model.equals("svm+text"))
			linkPred = new LinkPredictionWithSVMWithText(param.m_c, param.m_rho, param.m_lmTopK);
		
		linkPred.initMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		linkPred.getMMB().setR2TradeOffs(param.m_eta3, param.m_eta4);
		linkPred.getMMB().setsdB(param.m_sdB);
		linkPred.getMMB().setRho(param.m_rho);
		// commonly shared parameters.
		linkPred.getMMB().setR1TradeOffs(param.m_eta1, param.m_eta2);
		linkPred.getMMB().setM(param.m_M);
		linkPred.getMMB().setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
		linkPred.getMMB().setsdA(param.m_sdA);
		
		linkPred.getMMB().setBurnIn(param.m_burnin);
		linkPred.getMMB().setThinning(param.m_thinning);
		linkPred.getMMB().setNumberOfIterations(param.m_nuOfIterations);
	
		// training testing operations.
		linkPred.getMMB().loadLMFeatures(analyzer.getLMFeatures());
		linkPred.getMMB().loadUsers(analyzer.getUsers());
		linkPred.getMMB().setDisplayLv(displayLv);
		
		linkPred.getMMB().train();
		linkPred.linkPrediction();
		linkPred.calculateAllNDCGMAP();
		linkPred.calculateAvgNDCGMAP();
	}
}
