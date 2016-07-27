package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures.DPParameter;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;

public class MyDPExecution {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		DPParameter param = new DPParameter(args);

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		MTCLinAdaptWithDP adaptation = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setNumberOfIterations(param.m_nuOfIterations);
		adaptation.setAlpha(param.m_alpha);
		adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
		adaptation.setR2TradeOffs(param.m_eta3, param.m_eta4);
		adaptation.setM(param.m_M);
		adaptation.setsdA(param.m_sdA);
		adaptation.setsdB(param.m_sdB);
		adaptation.train();
		adaptation.test();
		adaptation.printInfo();
	}
}
