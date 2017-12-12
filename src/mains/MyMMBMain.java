package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedLMAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;


public class MyMMBMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;

		boolean enforceAdapt = true;
 
		String dataset = "YelpNew"; // "Amazon", "AmazonNew", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		int lmTopK = 1000; // topK for language model.
		int fvGroupSize = 800, fvGroupSizeSup = 5000;

		String fs = "DF";//"IG_CHI"
		
		String prefix = "./data/CoLinAdapt";
//		String prefix = "/zf8/lg5bt/DataSigir";

		String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset); // CV.
		String userFolder = String.format("%s/%s/Users_1000", prefix, dataset);
		String featureGroupFile = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSize);
		String featureGroupFileSup = String.format("%s/%s/CrossGroups_%d.txt", prefix, dataset, fvGroupSizeSup);
		String globalModel = String.format("%s/%s/GlobalWeights.txt", prefix, dataset);
		String lmFvFile = String.format("%s/%s/fv_lm_%s_%d.txt", prefix, dataset, fs, lmTopK);
		
		if(fvGroupSize == 5000 || fvGroupSize == 3071) featureGroupFile = null;
		if(fvGroupSizeSup == 5000 || fvGroupSizeSup == 3071) featureGroupFileSup = null;
		if(lmTopK == 5000 || lmTopK == 3071) lmFvFile = null;
		
		String friendFile = String.format("%s/%s/%sFriends_1000.txt", prefix, dataset, dataset);
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.buildFriendship(friendFile);
//		analyzer.checkFriendship();
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
//		PrintWriter writer = new PrintWriter(new File("yelp_features.txt"));
//		ArrayList<String> fvs = analyzer.getFeatures();
//		writer.write("Bias\n");
//		for(int i=0; i<fvs.size(); i++){
//			if(i != fvs.size()-1)
//				writer.write(fvs.get(i)+'\n');
//			else
//				writer.write(fvs.get(i));
//		}
//		writer.close();
		
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
//		mtsvm.printUserPerformance("./data/mtsvm_perf.txt");
		
//		// This part tries to pre-process the data in order to perform chi-square test.
//		Preprocess process = new Preprocess(analyzer.getUsers());
//		process.getRestaurantsStat();
//		process.printRestaurantStat("./data/yelp_chi_test.txt");
		
		// best parameter for yelp so far.
		double[] globalLM = analyzer.estimateGlobalLM();
		double alpha = 0.01, eta = 0.01, beta = 0.01;
		double sdA = 0.0425, sdB = 0.0425;
//		
////		MTCLinAdaptWithDP hdp = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup);
////		hdp.setAlpha(alpha);
////		
//		MTCLinAdaptWithHDP hdp = new MTCLinAdaptWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
//		hdp.setR2TradeOffs(eta3, eta4);
//		
//		hdp.setsdB(sdA);//0.2
//		hdp.setsdA(sdB);//0.2
//		
//		hdp.setR1TradeOffs(eta1, eta2);
//		hdp.setConcentrationParams(alpha, eta, beta);
//		
//		hdp.setBurnIn(10);
//		hdp.setNumberOfIterations(30);
//		
//		hdp.loadLMFeatures(analyzer.getLMFeatures());
//		hdp.loadUsers(analyzer.getUsers());
//		hdp.setDisplayLv(displayLv);
//		
//		hdp.train();
//		hdp.test();
//		
//		hdp.printUserPerformance("./data/yelp_hdp_perf.txt");
//		
//		CLRWithMMB mmb = new CLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setsdA(0.2);
//		
//		MTCLRWithMMB mmb = new MTCLRWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, globalLM);
//		mmb.setQ(0.1);
//		
//		CLinAdaptWithMMB mmb = new CLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, globalLM);
//		mmb.setsdB(0.1);

		MTCLinAdaptWithMMB mmb = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
		mmb.setR2TradeOffs(eta3, eta4);
		
		mmb.setsdA(sdA);
		mmb.setsdB(sdB);
				
		mmb.setR1TradeOffs(eta1, eta2);
		mmb.setConcentrationParams(alpha, eta, beta);

		mmb.setRho(0.01);
		mmb.setBurnIn(10);
//		mmb.setThinning(5);// default 3
		mmb.setNumberOfIterations(30);
		
		mmb.loadLMFeatures(analyzer.getLMFeatures());
		mmb.loadUsers(analyzer.getUsers());
		mmb.setDisplayLv(displayLv);					
		
		mmb.trainTrace(dataset+"_1000_trace.txt");
//		mmb.train();
//		mmb.test(); 
//		mmb.printEdgeCount("./edge_count.txt");
		
//		// Print out the current related models
//		long current = System.currentTimeMillis();
//		System.out.println(current);
//		String dir = String.format("./data/mmb/%d_%s", current, dataset);
//		mmb.saveEverything(dir);
		
	}
}
