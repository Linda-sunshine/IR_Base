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
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPConfidence;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPConfidenceInM;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDPExp;

public class MyDPExecution {
	
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
		
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, param.m_adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
//		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		CLRWithDP adaptation;
		double[] globalLM = analyzer.estimateGlobalLM();
		if(param.m_fv == 5000 || param.m_fv == 3071) featureGroupFile = null;
		if(param.m_fvSup == 5000 || param.m_fv == 3071) featureGroupFileSup = null;
		if(param.m_lmTopK == 5000 || param.m_lmTopK == 3071) lmFvFile = null;
		
		if(param.m_model.equals("mtclrdp")){
			adaptation = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
			adaptation.setQ(param.m_q);
		
		} else if(param.m_model.equals("clindp")){
			adaptation = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
			((CLinAdaptWithDP) adaptation).setsdB(param.m_sdB);

		} else if(param.m_model.equals("mtclindp")){
			adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
			adaptation.setLNormFlag(false);
			((MTCLinAdaptWithDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
	
		} else if(param.m_model.equals("mtclindpexp")){
			adaptation = new MTCLinAdaptWithDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
			adaptation.setLNormFlag(false);
			((MTCLinAdaptWithDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((MTCLinAdaptWithDPExp) adaptation).setBaseThreshold(param.m_base, param.m_th);

		} else if(param.m_model.equals("clrhdp")){
			adaptation = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLRWithHDP) adaptation).setC(param.m_c);
		
		} else if(param.m_model.equals("mtclrhdp")){
			adaptation = new MTCLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
			adaptation.setQ(param.m_q);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLRWithHDP) adaptation).setC(param.m_c);

		} else if(param.m_model.equals("clinhdp")){
			adaptation = new CLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
			((CLRWithHDP) adaptation).setC(param.m_c);

		} else if(param.m_model.equals("mtclinhdp")){
			adaptation = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((CLRWithHDP) adaptation).setC(param.m_c);

		} else if(param.m_model.equals("mtclinhdpexp")){
			adaptation = new MTCLinAdaptWithHDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((CLRWithHDP) adaptation).setC(param.m_c);
			((MTCLinAdaptWithHDPExp) adaptation).setPosteriorSanityCheck(param.m_post);
			((MTCLinAdaptWithHDPExp) adaptation).setThreshold(param.m_threshold);
		} else if(param.m_model.equals("mtclinhdpconf")){
			 adaptation = new MTCLinAdaptWithHDPConfidence(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((CLRWithHDP) adaptation).setC(param.m_c);
		} else if(param.m_model.equals("mtclinhdpconfm")){
			adaptation = new MTCLinAdaptWithHDPConfidenceInM(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null, globalLM);
			((CLRWithHDP) adaptation).setConcentrationParams(param.m_alpha, param.m_eta, param.m_beta);
			((CLinAdaptWithHDP) adaptation).setsdB(param.m_sdB);
			((MTCLinAdaptWithHDP) adaptation).setR2TradeOffs(param.m_eta3, param.m_eta4);
			((CLRWithHDP) adaptation).setC(param.m_c);
		}else{
			System.out.println("CLRWithDP is running....");
			adaptation = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		}
		
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
