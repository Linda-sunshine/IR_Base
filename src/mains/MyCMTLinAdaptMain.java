package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import clustering.KMeansAlg4Profile;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithClusters;

public class MyCMTLinAdaptMain {

	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 100;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
			
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
			
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
			
		int[] kmeanss = new int[]{1000};
		for(int kmeans: kmeanss){
		KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
		alg.train(analyzer.getUsers());
		
		// The returned clusters contain the corresponding cluster index of each user.
		int[] clusters = alg.getClusters();
		MTLinAdaptWithClusters adaptation = new MTLinAdaptWithClusters(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, featureGroupFileB, kmeans, clusters);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRsTradeOffs(lambda1, lambda2);
		adaptation.train();
		adaptation.test();
		
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		}
	}
}
