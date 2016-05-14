package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedUserAnalyzer;

import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncGeneral;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncMTLinAdapt;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;

public class MyAsyncMTLinAdaptMain {
		
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		// Best performance for mt-linadapt in amazon.
		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		// Best performance for mt-linadapt in yelp.
//		double eta1 = 0.9, eta2 =1 , lambda1 = 0.1, lambda2 = 0.1;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
			
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
				
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
			
		// Create an instance of asyncMTLinAdapt.
		asyncMTLinAdapt adaptation = new asyncMTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setTrainByUser(true);//train by review-driven mode.
		adaptation.setDisplayLv(displayLv);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setRsTradeOffs(lambda1, lambda2);
		adaptation.setRPTTime(3);
		adaptation.train();
		adaptation.test();
		
		/***
		// Create an instances of asyncLinAdapt model.
		asyncLinAdapt asynclinadapt = new asyncLinAdapt(classNumber,analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		asynclinadapt.loadUsers(analyzer.getUsers());
		asynclinadapt.setDataset(dataset);
		asynclinadapt.setDisplayLv(displayLv);
		asynclinadapt.train();
		asynclinadapt.test();		
		// Create an instances of asyncRegLR model.
		asyncRegLR asyncreglr = new asyncRegLR(classNumber,analyzer.getFeatureSize(), featureMap, globalModel);
		asyncreglr.loadUsers(analyzer.getUsers());
		asyncreglr.setDataset(dataset);
		asyncreglr.setDisplayLv(displayLv);
		asyncreglr.train();
		asyncreglr.test();
		
		// Create an instance of global model.
		asyncGeneral general = new asyncGeneral("global_all", classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
		general.setDataset(dataset);
		general.loadUsers(analyzer.getUsers());
		general.train();
		general.test();	
		***/
	}
}

