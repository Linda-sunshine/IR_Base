package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.semisupervised.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.MultiTaskSVM;
import opennlp.tools.util.InvalidFormatException;

public class LinAdaptMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		double trainRatio = 0, adaptRatio = 0.50;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 0.5, eta2 = 0.5, eta3 = 0.6, eta4 = 0.01, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = false;
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/SelectedVocab.csv"; // CV.
		String userFolder = "./data/CoLinAdapt/Users";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold,numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
//		//Create an instances of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);

//		//Create an instances of asyncLinAdapt model.
//		asyncLinAdapt adaptation = new asyncLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		
		//Create an instances of CoLinAdapt model.
		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
		
//		//Create an instances of zero-order asyncCoLinAdapt model.
//		asyncCoLinAdapt adaptation = new asyncCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);

		/** Added by lin for calling neighborhood learning.
		//The entrance for calling the CoLinAdaptWithNeighborhoodLearning.
		int fDim = 3; // xij contains <bias, bow, svd_sim>
		String svdFile = "./data/CoLinAdapt/Amazon_SVD_Scaled.mm";
		analyzer.loadSVDFile(svdFile);
		CoLinAdaptWithNeighborhoodLearning adaptation = new CoLinAdaptWithNeighborhoodLearning(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, fDim);
		*/
		
//		//Create an instances of first-order asyncCoLinAdapt model.
//		asyncCoLinAdaptFirstOrder adaptation = new asyncCoLinAdaptFirstOrder(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, neighborsHistoryWeight);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
//		adaptation.setTestMode(TestMode.TM_batch);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);
		
		adaptation.train();
		adaptation.test();
		
		//Create the instance of MT-SVM
		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize(), analyzer.getUsers());
		mtsvm.setBias(true);
		mtsvm.train();
		mtsvm.test();
	}
}
