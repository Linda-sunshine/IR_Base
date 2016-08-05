package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncMTLinAdapt;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;

public class MyMTLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 100;
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
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", dataset, 800);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", dataset, 800);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);

//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//	
//		String[] opts = new String[]{"G", "D", "G+D"};
//		for(String i: opts){
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
//		analyzer.loadCategory("./data/category.txt");
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
				
		//Create an instance of MTLinAdapt with Super user sharing different dimensions.
		MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null); 
		mtlinadaptsup.loadUsers(analyzer.getUsers());
		mtlinadaptsup.setDisplayLv(displayLv);
		mtlinadaptsup.setR1TradeOffs(eta1, eta2);
		mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
		mtlinadaptsup.train();
		mtlinadaptsup.test();
		mtlinadaptsup.saveModel("./data/models/mtlinadapt_1");
		mtlinadaptsup.saveSupModel("./data/mtlinadapt_supmodel.txt");
	}
}
