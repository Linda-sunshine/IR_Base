package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;

public class MyLinAdpatTestMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 200;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		double eta1 = 1, eta2 = 0.5, eta3 = 0.1, eta4 = 0.1;

		double eta1 = 1, eta2 = 0.5, eta3 = 0.5, eta4 = 0.5, neighborsHistoryWeight = 0.5;
//		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
			
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
//		analyzer.loadCategory("category.txt");
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
//		analyzer.loadUserWeights("./data/models/mtsvm_0.5/", "classifer");
		
		int kFold = 100, kMeans = 400;
		CrossFeatureSelection fs = new CrossFeatureSelection(analyzer.getCorpus(), classNumber, analyzer.getFeatureSize(), kFold, kMeans);
		fs.train();
		fs.kMeans();
		fs.writeResults();
		
		// Create an instance of CoLinAdapt model.
//		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		
//		adaptation.loadUsers(analyzer.getUsers());
//		double sumR2 = adaptation.accumulateR2();
//		System.out.println("Sum R2: " + sumR2);
		
//		int time = 20;
//		double[] R2s = new double[time];
//		for(int i=0; i<time; i++){
//			CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//			adaptation.loadUsers(analyzer.getUsers());
//			R2s[i] = adaptation.accumulateR2();
//		}
//		for(double r2: R2s)
//			System.out.println(r2);
	}
}
