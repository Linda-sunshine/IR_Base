package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.WeightedAvgAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.WeightedAvgTransAdapt;

public class MyAvgAdaptMain {

	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int[] topKNeighbors = new int[]{50, 100, 200, 400, 800, 1600, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 0.5, eta2 = 0.5;

//		double eta1 = 1, eta2 = 0.5, eta3 = 0.5, eta4 = 0.5, neighborsHistoryWeight = 0.5;
//		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		for(int i: topKNeighbors){
		
		WeightedAvgTransAdapt adaptation = new WeightedAvgTransAdapt(classNumber, analyzer.getFeatureSize(), featureMap, i, globalModel, featureGroupFile);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.train();
		adaptation.test();
		
		// Clear the performance.
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		}
	}
}
