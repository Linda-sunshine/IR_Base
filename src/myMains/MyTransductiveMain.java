package myMains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.semisupervised.GaussianFieldsByRandomWalkWithFriends;

public class MyTransductiveMain {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		/***parameters used in loading users***/
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
		
//		String prefix = "./data/CoLinAdapt";
		String prefix = "/zf8/lg5bt/DataSigir";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users", prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildTrainFriendship(friendFile);
//		analyzer.checkFriendship();
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
		/**parameters used in random walk**/
		System.out.println("Start Transductive Learning, wait...");
		double learningRatio = 1.0;
		int k = 30, kPrime = 20; // k nearest labeled, k' nearest unlabeled
		double tAlpha = 1.0, tBeta = 0.1; // labeled data weight, unlabeled data weight
		double tDelta = 1e-5, tEta = 0.6; // convergence of random walk, weight of random walk
		boolean simFlag = false, weightedAvg = true;
		int bound = 0; // bound for generating rating constraints (must be zero in binary case)
		int topK = 25; // top K similar documents for constructing p
		
		String multipleLearner = "SVM";
		double C = 1.0;
		_Corpus c = analyzer.getCorpus();
		GaussianFieldsByRandomWalkWithFriends walk = new GaussianFieldsByRandomWalkWithFriends(c, multipleLearner, C,
			learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg);
		walk.constructTrainTestDocs(analyzer.getUsers());
//		walk.setFriendship(analyzer.getFriendship());
		walk.train();
		walk.test();
		walk.printUserPerformance("./data/random_walk_perf.txt");
	}
}
