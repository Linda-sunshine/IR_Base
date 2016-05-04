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
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		// Best performance for CoLinAdapt.
//		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859;

		// Best performance for mt-linadapt in amazon.
//		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		double eta3 = 0.1, eta4 = 0.3;
//		double eta1 = 1, eta2 = 0.4, lambda1 = 0.1, lambda2 = 0.7;
		// Best performance for mt-linadapt in yelp.
		double eta1 = 0.9, eta2 =1 , lambda1 = 0.1, lambda2 = 0.1;
		boolean enforceAdapt = true;
		double[] perf;
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

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
			
		//Create an instance of MTLinAdapt with Super user sharing different dimensions.
//		MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null); 
//		mtlinadaptsup.loadUsers(analyzer.getUsers());
//		mtlinadaptsup.setDisplayLv(displayLv);
//		mtlinadaptsup.setR1TradeOffs(eta1, eta2);
//		mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
//		mtlinadaptsup.train();
//		mtlinadaptsup.test();
//		writePerformance("MTLinAdapt_perf.txt", mtlinadaptsup.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		// Create the instance of global SVM.
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.setBias(false);
//		gsvm.train();
//		gsvm.test();
//		writePerformance("GlobalSVM_perf_falsebias.txt", gsvm.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.setBias(true);
//		gsvm.train();
//		gsvm.test();
//		writePerformance("GlobalSVM_perf_truebias.txt", gsvm.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		// Create the instance of individual SVM.
//		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.setBias(false);
//		indsvm.train();
//		indsvm.test();
//		writePerformance("IndividualSVM_perf.txt", indsvm.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		// Create an intance of RegLR.
//		RegLR reglr = new RegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		reglr.loadUsers(analyzer.getUsers());
//		reglr.setDisplayLv(displayLv);
//		reglr.train();
//		reglr.test();
//		writePerformance("RegLR_perf.txt", reglr.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		// Create an instances of LinAdapt model.
//		LinAdapt adaptation = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.train();
//		adaptation.test();
//		writePerformance("Base_perf.txt", adaptation.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
////		mtsvm.setBias(false);
//		mtsvm.train();
//		mtsvm.test();
//		writePerformance("MTSVM_perf.txt", mtsvm.getUserList());
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
	}
	
	// Print out the performance of users.
	public static void writePerformance(String filename, ArrayList<_AdaptStruct> users){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(_AdaptStruct u: users)
				writer.write(String.format("%s,%d,%.4f,%.4f\n", u.getUserID(), u.getReviews().size(), u.getPerfStat().getF1(0), u.getPerfStat().getF1(1)));
			writer.close();	
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}
