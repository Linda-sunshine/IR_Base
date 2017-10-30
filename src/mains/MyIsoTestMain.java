package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.IsoUserAnalyzer;
import Classifier.supervised.IndSVMColdStart;
import Classifier.supervised.modelAdaptation.DirichletProcess.IsoMTCLinAdaptWithDP;

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
			boolean enforceAdapt = true;

			String dataset = "YelpNew"; // "Amazon", "Yelp"
			String tokenModel = "./data/Model/en-token.bin"; // Token model.
			
			String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
			String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
			String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
			String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
			String dir = String.format("/home/lin/DataWsdm2017/%s/%s_", dataset, dataset);

//			String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//			String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//			String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//			String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
			for(int userTestThreshold: testUsers){

			IsoUserAnalyzer analyzer = new IsoUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, false);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.setUserTestThreshold(userTestThreshold);//userTestThreshold); // has to be set before load user dir.
			analyzer.loadUserDir(userFolder);
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			analyzer.constructSparseVector4Users(); // The profiles are based on the TF-IDF with different DF schemes.
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

			//Yelp best parameter: 0.23 0.1 0.04 0.01
			double sdA = 0.2, sdB =0.2;
			
			//Amazon parameters.
//			double eta1 = 0.01, eta2 = 0.01, eta3 = 0.06, eta4 = 0.01;
//			double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;

			//Yelp parameters.
			double eta1 = 0.09, eta2 = 0.02, eta3 = 0.07, eta4 = 0.03;
			String[] models = new String[]{"mtclindp", "clrdp", "mtclrdp", "clindp", "mtsvm", "linadapt", "indsvm"};
			double[][] perf = new double[models.length][6];

			int[] ks = new int[]{1, 2, 3};
			
			for(int k: ks){
			System.out.print(String.format("***********test user size: %d, nu of rvw for cluster assignment: %d********\n", userTestThreshold, k));

			/***our algorithm: MTCLinAdaptWithDP***/
			IsoMTCLinAdaptWithDP adaptation = new IsoMTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
			adaptation.loadUsers(analyzer.getUsers());
			adaptation.setDisplayLv(displayLv);
			adaptation.setLNormFlag(false);
			adaptation.setClusterAssignThreshold(k);
			adaptation.setsdA(sdA);
			adaptation.setsdB(sdB);
			adaptation.setAlpha(1);
			adaptation.setNumberOfIterations(5);
			adaptation.setR1TradeOffs(eta1, eta2);
			adaptation.setR2TradeOffs(eta3, eta4);
			adaptation.train();
			adaptation.test();
			adaptation.printInfo();
			perf[0] = adaptation.getPerf();
			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
//			
//			//clrdp
//			IsoCLRWithDP clrdp = new IsoCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//			clrdp.loadUsers(analyzer.getUsers());
//			clrdp.setDisplayLv(displayLv);
//			clrdp.setLNormFlag(false);
//			clrdp.setClusterAssignThreshold(k);
//			clrdp.setsdA(sdA);
//			clrdp.setAlpha(1);
//			clrdp.setR1TradeOffs(eta1, eta2);
//			clrdp.train();
//			clrdp.test();
//			perf[1] = clrdp.getPerf();
//
//			for(_User u: analyzer.getUsers())
//				u.getPerfStat().clear();
//			
//			//mtclrdp
//			IsoMTCLRWithDP mtclrdp = new IsoMTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//			mtclrdp.loadUsers(analyzer.getUsers());
//			mtclrdp.setDisplayLv(displayLv);
//			mtclrdp.setLNormFlag(false);
//			mtclrdp.setQ(0.4);
//			mtclrdp.setClusterAssignThreshold(k);
//			mtclrdp.setsdA(sdA);
//			mtclrdp.setAlpha(1);
//			mtclrdp.setR1TradeOffs(eta1, eta2);
//			mtclrdp.train();
//			mtclrdp.test();
//			perf[2] = mtclrdp.getPerf();
//
//			for(_User u: analyzer.getUsers())
//				u.getPerfStat().clear();
//			
//			//clindp
//			IsoCLinAdaptWithDP clindp = new IsoCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//			clindp.loadUsers(analyzer.getUsers());
//			clindp.setDisplayLv(displayLv);
//			clindp.setLNormFlag(false);
//			clindp.setClusterAssignThreshold(k);
//			clindp.setsdA(sdA);
//			clindp.setsdB(sdB);
//			clindp.setAlpha(1);
//			clindp.setR1TradeOffs(eta1, eta2);
//			clindp.train();
//			clindp.test();
//			perf[3] = clindp.getPerf();
//
//			for(_User u: analyzer.getUsers())
//				u.getPerfStat().clear();
//			
//			//mtsvm
//			MTSVMColdStart mtsvm = new MTSVMColdStart(classNumber, analyzer.getFeatureSize(), k);
//			mtsvm.loadUsers(analyzer.getUsers());
//			mtsvm.train();
//			mtsvm.test();
//			perf[4] = mtsvm.getPerf();
//
//			for(_User u: analyzer.getUsers())
//				u.getPerfStat().clear();
//			
//			//linadapt
//			LinAdaptColdStart linadapt = new LinAdaptColdStart(classNumber, analyzer.getFeatureSize(),featureMap, globalModel, featureGroupFile, k);
//			linadapt.loadUsers(analyzer.getUsers());
//			linadapt.setLNormFlag(false);
//			linadapt.train();
//			linadapt.test();
//			perf[5] = linadapt.getPerf();
//
//			for(_User u: analyzer.getUsers())
//				u.getPerfStat().clear();
			
			//individual svm
			IndSVMColdStart indsvm = new IndSVMColdStart(classNumber, analyzer.getFeatureSize(), k);
			indsvm.loadUsers(analyzer.getUsers());
			indsvm.train();
			indsvm.test();
			perf[6] = indsvm.getPerf();

			for(_User u: analyzer.getUsers())
				u.getPerfStat().clear();
			
			PrintWriter writer;
			try{
				String filename = String.format("./data/%d_%d_cold.txt", userTestThreshold, k);
				writer = new PrintWriter(new File(filename));
				for(int i=0; i<models.length; i++){
					writer.write(String.format("%s\t%.4f\t%.4f\n", models[i], perf[i][0], perf[i][1]));
				}
				writer.close();
			} catch (IOException e){
				e.printStackTrace();
			}
			}}
		}
}
