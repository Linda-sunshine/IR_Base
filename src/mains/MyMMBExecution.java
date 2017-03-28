package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures.DPParameter;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDPExp;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.CLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPExp;
import Classifier.supervised.modelAdaptation.MMB.CLRWithMMB;

public class MyMMBExecution {
	
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

		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String providedCV = String.format("%s/%s/SelectedVocab.csv", param.m_prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users", param.m_prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, dataset, param.m_fv);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", param.m_prefix, dataset, param.m_fvSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", param.m_prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", param.m_prefix, dataset, param.m_fs, param.m_lmTopK);
				
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/YelpNew/yelpFriends.txt", param.m_prefix);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, param.m_adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		double[] globalLM = analyzer.estimateGlobalLM();
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		CLRWithMMB adaptation = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
		
		// commonly shared parameters.
		adaptation.setM(param.m_M);
		adaptation.setAlpha(param.m_alpha);
		adaptation.setsdA(param.m_sdA);
		adaptation.setThinning(param.m_thinning);
		adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
		adaptation.setNumberOfIterations(param.m_nuOfIterations);

		// training testing operations.
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.train();
		adaptation.test();
	}
}
