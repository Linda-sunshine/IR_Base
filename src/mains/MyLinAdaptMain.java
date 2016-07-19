package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
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
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLogisticRegressionWithDP;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;

public class MyLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
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
		
//		double[] alphas = new double[]{0.1, 0.5, 1, 2, 3, 4, 5};
//		for(double alpha: alphas){
		//CLogisticRegressionWithDP
		CLinAdaptWithDP adaptation = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileB);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setBurnIn(30);
		adaptation.setM(20);
		adaptation.setAlpha(1);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRsTradeOffs(eta3, eta4);
		adaptation.EM();
		adaptation.test();
		adaptation.printInfo();
//		}
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

		// Create an instance of ClusterLinAdapt.
//		ClusteredLinAdapt adaptation = new ClusteredLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile, kmeans, clusters);
//		adaptation.setParameters(1, 1, 1);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setRcRgTradeOffs(eta3, eta4);
//		adaptation.train();
//		adaptation.test();
//		
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//			
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
//		
//		for(_User u: analyzer.getUsers())
//		u.getPerfStat().clear();
//	
//		//MultiTaskSVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.setBias(true);
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
	}
}
