package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures.DPParameter;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB4LinkPrediction;
import Classifier.supervised.modelAdaptation.MMB.SVMBasedLinkPrediction;

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
		String trainFolder = String.format("%s/%s/Users_%d", param.m_prefix, param.m_data, param.m_trainSize);
		String testFolder = String.format("%s/%s/Users_%d", param.m_prefix, param.m_data, param.m_testSize);
		
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fv);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, param.m_data, param.m_fvSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", param.m_prefix, param.m_data);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", param.m_prefix, param.m_data, param.m_fs, param.m_lmTopK);
				
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", param.m_prefix,param.m_data,param.m_data);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, 1, true);
		
		// load training users with (adaptRatio=1, testRatio=0)
		analyzer.loadUserDir(trainFolder);
				
		// load testing users with (adaptaRatio=0, testRatio=1)
		analyzer.config(trainRatio, 0, false);
		analyzer.loadUserDir(testFolder);
		
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		double[] globalLM = analyzer.estimateGlobalLM();
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		MTCLinAdaptWithMMB4LinkPrediction adaptation = null;
		
		
		if(param.m_model.equals("svm")){
			adaptation = new SVMBasedLinkPrediction(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		} else if(param.m_model.equals("mmb")){
			adaptation = new MTCLinAdaptWithMMB4LinkPrediction(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
			
		}
		
		adaptation.setR2TradeOffs(param.m_eta3, param.m_eta4);
		adaptation.setsdB(param.m_sdB);
		adaptation.setRho(param.m_rho);
		// commonly shared parameters.
		adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
		adaptation.setM(param.m_M);
		adaptation.setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
		adaptation.setsdA(param.m_sdA);
		
		adaptation.setBurnIn(param.m_burnin);
		adaptation.setThinning(param.m_thinning);
		adaptation.setNumberOfIterations(param.m_nuOfIterations);
	
		// training testing operations.
		adaptation.loadLMFeatures(analyzer.getLMFeatures());
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.calculateFrdStat();

		adaptation.checkTestReviewSize();
		adaptation.setDisplayLv(displayLv);
		
		adaptation.train();
		System.out.println("[Info]Finish model training and start link prediction...");
		
		if(param.m_linkMulti)
			adaptation.linkPrediction_MultiThread();
		else
			adaptation.linkPrediction();
		adaptation.printLinkPrediction(param.m_dir);
	}
}
