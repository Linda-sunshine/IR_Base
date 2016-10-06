package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.IsoUserAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.IndSVMColdStart;
import Classifier.supervised.modelAdaptation.MTSVMColdStart;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdaptColdStart;
import Classifier.supervised.modelAdaptation.DirichletProcess.IsoMTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;

public class MyIsoTestMain {

	//In the main function, we want to input the data and do adaptation 
		public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

			int classNumber = 2;
			int Ngram = 2; // The default value is unigram.
			int lengthThreshold = 5; // Document length threshold
			double trainRatio = 0, adaptRatio = 1;
			int[] testUsers = new int[]{2000, 3000, 4000, 5000};
			int displayLv = 1;
			int numberOfCores = Runtime.getRuntime().availableProcessors();

			double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
			boolean enforceAdapt = true;

			String dataset = "Yelp"; // "Amazon", "Yelp"
			String tokenModel = "./data/Model/en-token.bin"; // Token model.
			
//			String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//			String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//			String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//			String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
//			String dir = String.format("/home/lin/DataWsdm2017/%s/%s_", dataset, dataset);

			String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
			String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
			String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
			String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
			String dir = String.format("/if15/lg5bt/DataWsdm2017/%s/%s_", dataset, dataset);
			for(int userTestThreshold: testUsers){

			IsoUserAnalyzer analyzer = new IsoUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.setUserTestThreshold(userTestThreshold);//userTestThreshold); // has to be set before load user dir.
			analyzer.loadUserDir(userFolder);
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

			//Yelp best parameter: 0.23 0.1 0.04 0.01
			double sdA = 0.23, sdB = 0.1; eta1 = 0.04; eta3 = 0.01; 
//			double sdA = 0.4, sdB = 0.2; eta1 = 0.005; eta3 = 0.005; 
			int[] ks = new int[]{1, 2, 3};
			for(int k: ks){
			/***our algorithm: MTCLinAdaptWithDP***/
			IsoMTCLinAdaptWithDP adaptation = new IsoMTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
			adaptation.loadUsers(analyzer.getUsers());
			adaptation.setDisplayLv(displayLv);
			adaptation.setLNormFlag(false);
			adaptation.setClusterAssignThreshold(k);
//			adaptation.setAllFlag(true);
			adaptation.setsdA(sdA);
			adaptation.setsdB(sdB);
			adaptation.setAlpha(1);
			adaptation.setR1TradeOffs(eta1, eta1);
			adaptation.setR2TradeOffs(eta3, eta3);
			adaptation.train();
			adaptation.test();
			adaptation.printInfo();
			System.out.print(String.format("[Oracle]test user size: %d, nu of rvw for cluster assignment: %d\n********\n", userTestThreshold, k));
//			adaptation.printUserPerf(dir+"mtclindp.txt");
//			adaptation.saveClusterModel(dir + "mtclindp_c_0.5/");
//			adaptation.saveModel(dir + "mtclindp_u_0.5");
//			adaptation.saveClusterInfo(dir + "clusterInfo.txt");
			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
			
			//mtsvm
			MTSVMColdStart mtsvm = new MTSVMColdStart(classNumber, analyzer.getFeatureSize(), k);
			mtsvm.loadUsers(analyzer.getUsers());
			mtsvm.train();
			mtsvm.test();
			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
			
			//linadapt
			LinAdaptColdStart linadapt = new LinAdaptColdStart(classNumber, analyzer.getFeatureSize(),featureMap, globalModel, featureGroupFile, k);
			linadapt.loadUsers(analyzer.getUsers());
			linadapt.setLNormFlag(false);
			linadapt.train();
			linadapt.test();
			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
			
			//individual svm
			IndSVMColdStart indsvm = new IndSVMColdStart(classNumber, analyzer.getFeatureSize(), k);
			indsvm.loadUsers(analyzer.getUsers());
			indsvm.train();
			indsvm.test();
			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
			}}
		}
}
