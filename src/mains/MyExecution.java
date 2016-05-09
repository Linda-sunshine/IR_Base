package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdaptWithR2OverWeights;
import structures.CoLinAdaptParameter;

public class MyExecution {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		CoLinAdaptParameter param = new CoLinAdaptParameter(args);
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 200;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;
		
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", param.m_data);
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", param.m_data); // CV.
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", param.m_data);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", param.m_data);
		
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadCategory("category.txt");
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		CoLinAdaptWithR2OverWeights adaptation = new CoLinAdaptWithR2OverWeights(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(param.m_eta1, param.m_eta2);
		adaptation.setR2TradeOffs(param.m_eta3, param.m_eta4);
		adaptation.train();
		adaptation.test();
	}
}
