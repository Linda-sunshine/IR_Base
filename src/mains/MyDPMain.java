package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import clustering.KMeansAlg4Profile;
import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.ReTrain;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.CLinAdaptWithKmeans;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDPExp;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.CLinAdaptWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;
import Classifier.supervised.modelAdaptation.HDP.MTCLinAdaptWithHDP;

public class MyDPMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
	
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp", "YelpNew"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800" + ".txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_1600.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		String lmFvFile = String.format("./data/CoLinAdapt/%s/fv_lm.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String featureFile4LM = String.format("/if15/lg5bt/DataSigir/%s/fv_lm.txt", dataset);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, null, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
				
//		int max = 0;
//		for(_User u: analyzer.getUsers()){
//			if(u.getReviewSize() > max)
//				max = u.getReviewSize();
//		}
//		System.out.println(max);
//		
//		PrintWriter writer = new PrintWriter(new File("size.txt"));
//		for(_User u: analyzer.getUsers())
//			writer.write(u.getUserID()+"\t"+u.getReviewSize()+"\n");
//		writer.close();
////		analyzer.getStat();
		double sdA = 0.2, sdB =0.2;
		
		//Amazon parameters.
		double eta1 = 0.06, eta2 = 0.01, eta3 = 0.06, eta4 = 0.01;
//		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;

		//Yelp parameters.
//		double eta1 = 0.09, eta2 = 0.02, eta3 = 0.07, eta4 = 0.03;

//		/***baseline 0: base***/
//		Base base = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		base.loadUsers(analyzer.getUsers());
//		base.setPersonalizedModel();
//		base.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 1: global***/
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.train();
//		gsvm.test();
//		gsvm.savePerf("./data/gsvm_perf.txt");
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 2: individual svm***/
//		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		indsvm.loadUsers(analyzer.getUsers());
//		indsvm.train();
//		indsvm.test();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 3: LinAdapt***/
//		//Create an instances of LinAdapt model.
//		LinAdapt linadapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel,featureGroupFile);
//		linadapt.loadUsers(analyzer.getUsers());
//		linadapt.setDisplayLv(displayLv);
//		linadapt.setR1TradeOffs(eta1, eta1);
//		linadapt.train();
//		linadapt.test();
//		linadapt.saveModel(String.format("./data/%s_linadapt_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//		u.getPerfStat().clear();
//
//		/***baseline 4: CLinAdaptWithKmeans***/
//		// We perform kmeans over user weights learned from individual svms.
//		int kmeans = 200;
//		int[] clusters;
//		KMeansAlg4Profile alg = new KMeansAlg4Profile(classNumber, analyzer.getFeatureSize(), kmeans);
//		alg.train(analyzer.getUsers());
//		clusters = alg.getClusters();// The returned clusters contain the corresponding cluster index of each user.
//		
//		CLinAdaptWithKmeans clinkmeans = new CLinAdaptWithKmeans(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, kmeans, clusters);
//		clinkmeans.loadUsers(analyzer.getUsers());
//		clinkmeans.setDisplayLv(displayLv);
//		clinkmeans.setLNormFlag(false);
//		clinkmeans.setR1TradeOffs(eta1, eta1);
//		clinkmeans.train();
//		clinkmeans.test();
//		clinkmeans.setParameters(0, 1, 1);
////		clinkmeans.saveModel(String.format("./data/%s_clinkmeans_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
//		/***baseline 5: CLinAdaptWithDP***/
//		// Create an instance of CLinAdaptWithDP
//		CLinAdaptWithDP clindp = new CLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
//		clindp.loadUsers(analyzer.getUsers());
//		clindp.setDisplayLv(displayLv);
//		clindp.setLNormFlag(false);
//		clindp.setR1TradeOffs(eta1, eta1);
//		clindp.setsdA(sdA);
//		clindp.setsdB(sdB);
//		clindp.train();
//		clindp.test();
////		clindp.saveModel(String.format("./data/%s_clindp_0.5_1", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 6: CLRWithDP***/
//		CLRWithDP clrdp = new CLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
//		clrdp.loadUsers(analyzer.getUsers());
//		clrdp.setDisplayLv(displayLv);
//		clrdp.setLNormFlag(false);
//		clrdp.setR1TradeOffs(eta1, eta1);
//		clrdp.train();
//		clrdp.test();
//		clrdp.saveModel(String.format("./data/%s_clrdp_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//		
//		/***baseline 7: mtsvm***/
//		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.savePerf("./data/mtsvm_perf.txt");
//		mtsvm.saveModel(String.format("./data/%s_mtsvm_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
//		/***baseline 8: MTCLRWithDP***/
//		// Create an instance of MTCLRWithDP
//		MTCLRWithDP mtclrdp = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);	
//		mtclrdp.loadUsers(analyzer.getUsers());
//		mtclrdp.setDisplayLv(displayLv);
//		mtclrdp.setLNormFlag(false);
//		mtclrdp.setQ(0.4);
//		mtclrdp.setsdA(sdA);
//		mtclrdp.setR1TradeOffs(eta1, eta2);
//		mtclrdp.train();
//		mtclrdp.test();
//		mtclrdp.savePerf("./data/mtclrdp_perf.txt");
//		mtclrdp.saveModel(String.format("./data/%s_mtclrdp_0.5_1/", dataset));
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
//
		/***our algorithm: MTCLinAdaptWithDP***/
		MTCLinAdaptWithDPExp adaptation = new MTCLinAdaptWithDPExp(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.setLNormFlag(false);
		adaptation.setsdA(sdA);
		adaptation.setsdB(sdB);
		adaptation.setR1TradeOffs(eta1, eta2);
		adaptation.setR2TradeOffs(eta3, eta4);
		//String traceFile = dataset + "_iter.csv";
		//adaptation.trainTrace(traceFile);
		//adaptation.setNumberOfIterations(100);
		
		adaptation.train();
		adaptation.test();
		
		int threshold = 100;
		adaptation.CrossValidation(5, threshold);
			
//		long time = System.currentTimeMillis();
//		String pattern = dataset+"_"+time;
////		String umodel = String.format("./data/%s/%s_mtclindp_u_0.5/", pattern, dataset);
////		String cmodel = String.format("./data/%s/%s_mtclindp_c_0.5/", pattern, dataset);
//		String perf = String.format("./data/mtclindp_perf.txt", pattern, dataset);
////		adaptation.saveModel(umodel);
////		adaptation.saveClusterModels(cmodel);
//		adaptation.savePerf(perf);
//	
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
	}
}
