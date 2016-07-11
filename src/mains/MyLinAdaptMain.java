package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import clustering.KMeansAlg4Profile;
import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._User;
import utils.Utils;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.CoLinAdaptWithR1OverWeights;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.ClusteredLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdaptWithDiffFeatureGroups;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdaptWithR2OverWeights;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdaptOverall;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;

public class MyLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 100;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 0.1, eta2 = 0.1, eta3 = 0.1, eta4 = 0.1;

//		double eta1 = 1, eta2 = 0.5, eta3 = 0.5, eta4 = 0.5, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users_1000", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		//String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
//		analyzer.loadCategory("./data/category.txt");
//		analyzer.setCtgThreshold(ctgCount);
		analyzer.loadUserDir(userFolder);
		// The second parameter stands for DF scheme, "G"-global, "D"-adaptation, "G+D"-both
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		// Load svd of each user.
//		String svdFile = "./data/CoLinAdapt/Amazon/Amazon_SVD.mm";
//		analyzer.loadSVDFile(svdFile);
		
		// Create an instances of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.train();
//		adaptation.test();
//		 
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		 
//		 // Create an instances of asyncLinAdapt model.
//		 asyncLinAdapt adaptation = new asyncLinAdapt(classNumber,
//		 analyzer.getFeatureSize(), featureMap, globalModel,
//		 featureGroupFile);
		
//		// Create an instance of CoLinAdaptWithNeighborhoodLearning model.
//		int fDim = 3; // xij contains <bias, bow, svd_sim>
//		CoLinAdaptWithNeighborhoodLearning adaptation = new CoLinAdaptWithNeighborhoodLearning(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, fDim);
		
		// Create an instance of zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt adaptation = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

//		//Create an instance of first-order asyncCoLinAdapt model.
//		asyncCoLinAdaptFirstOrder adaptation = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, neighborsHistoryWeight);
//		int[] kFolds = new int[]{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
//		int[] kMeans = new int[]{200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1300, 1400, 1500, 1600};
//		for(int kFold: kFolds){
//			for(int kMean: kMeans){
//		
//		CrossFeatureSelection fs = new CrossFeatureSelection(analyzer.getCorpus(), classNumber, analyzer.getFeatureSize(), kFold, kMean);
//		fs.train();
//		fs.kMeans();
//		
//		String featureGroupFile = fs.getFilename();
		// Create an instance of CoLinAdapt model.
//		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
		// Create an instance of CoLinAdaptWithR1OverWeights.
//		CoLinAdaptWithR1OverWeights adaptation = new CoLinAdaptWithR1OverWeights(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
	
		// Create an instance of CoLinAdaptWithR2OverWeights.
//		CoLinAdaptWithR2OverWeights adaptation = new CoLinAdaptWithR2OverWeights(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
	
		// Create an instance of CoLinAdapt with different feature groups for different classes.
//		CoLinAdaptWithDiffFeatureGroups adaptation = new CoLinAdaptWithDiffFeatureGroups(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, featureGroupFileB);
		
		// Create an instance of MTCoLinAdapt model.
//		MTCoLinAdapt adaptation = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		// Create an instance of LinAdaptOverall.
//		LinAdaptOverall adaptation = new LinAdaptOverall(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		
//		MTRegLR adaptation = new MTRegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		
		int kmeans = 10;
		int[] clusters;
		KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
		alg.train(analyzer.getUsers());
		clusters = alg.getClusters();
		
//		double[] cs = new double[]{0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
//		for(double c: cs){
		// Create an instance of ClusterLinAdapt.
		ClusteredLinAdapt adaptation = new ClusteredLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile, kmeans, clusters);
		adaptation.setParameters(1, 1, 1);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRcRgTradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();
		
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
		adaptation = new ClusteredLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile, kmeans, clusters);
		adaptation.setParameters(1, 1, 0);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRcRgTradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();
	
		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
		adaptation = new ClusteredLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile, kmeans, clusters);
		adaptation.setParameters(1, 0, 1);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRcRgTradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();

		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
		adaptation = new ClusteredLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile, kmeans, clusters);
		adaptation.setParameters(0, 1, 1);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRcRgTradeOffs(eta3, eta4);
		adaptation.train();
		adaptation.test();
		
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
		
//		for(_User u: analyzer.getUsers())
//		u.getPerfStat().clear();
	
//		//MultiTaskSVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.setBias(true);
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
	}
}
