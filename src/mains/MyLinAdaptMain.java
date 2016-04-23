package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._User;
import utils.Utils;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;

public class MyLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		double eta1 = 1, eta2 = 0.5, eta3 = 0.1, eta4 = 0.03;

//		double eta1 = 1, eta2 = 0.5, eta3 = 1, eta4 = 0.03, neighborsHistoryWeight = 0.5;
		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
			
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		// Load svd of each user.
//		String svdFile = "./data/CoLinAdapt/Amazon/Amazon_SVD.mm";
//		analyzer.loadSVDFile(svdFile);
		
//		 // Create an instances of LinAdapt model.
//		 LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);

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
		
		// Create an instance of CoLinAdapt model.
		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		// Create an instance of MTCoLinAdapt model.
//		MTCoLinAdapt adaptation = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		//adaptation.setTestMode(TestMode.TM_batch);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);

		adaptation.train();
		adaptation.test();

		//adaptation.saveModel("data/results/colinadapt");

		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
	
	}
}
