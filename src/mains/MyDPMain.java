package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import clustering.KMeansAlg;
import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._stat;
import utils.Utils;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.ReTrain;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLRWithDP;
import Classifier.supervised.modelAdaptation.HDP.CLRWithHDP;

public class MyDPMain {
	
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

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users_1000", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		String dir = "./data/";

//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users_1000", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String dir = String.format("/if15/lg5bt/DataWsdm2017/%s/%s_", dataset, dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

//		KMeansAlg kmeans = new KMeansAlg(classNumber, analyzer.getFeatureSize(), 40);
//		ArrayList<_Review> mergeDocs = analyzer.mergeRvws();
//		
//		HashMap<String, Integer> ctgIndex = new HashMap<String, Integer>();
//		int index = 0;
//		for(String c: analyzer.getCategories())
//			ctgIndex.put(c, index++);
//		
//		
//		kmeans.trainKmeans(mergeDocs);
//		int k = 40;
//		String filename = String.format("./data/kmeans_%d.xls", k);
//		kmeans.writeRatioes(mergeDocs, filename, ctgIndex);
		
//		MTCLRWithDP mtclrdp = new MTCLRWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);	
//		mtclrdp.loadUsers(analyzer.getUsers());
//		mtclrdp.setDisplayLv(displayLv);
//		mtclrdp.setLNormFlag(false);
//		mtclrdp.setQ(0.4);
//		mtclrdp.setR1TradeOffs(eta1, eta2);
//		mtclrdp.train();
//		mtclrdp.test();
//		mtclrdp.printInfo();
//		for(_User u: analyzer.getUsers())
//			u.getPerfStat().clear();
		
		double[] globalLM = analyzer.estimateGlobalLM();
//		double[] test = new double[analyzer.getFeatureSize()];
//		for(_User u: analyzer.getUsers()){
//			for(_Review r: u.getReviews()){
//				for(_SparseFeature f: r.getSparse()){
//					test[f.getIndex()] += f.getTF();
//				}
//			}
//		}
//		int count = 0;
//		for(int i=0; i<test.length; i++){
//			if(test[i] != globalLM[i])
//				System.out.println(i+"\t"+test[i]+"!="+globalLM[i]+"\t"+count++);
//		}
		CLRWithHDP clrhdp = new CLRWithHDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
		clrhdp.setGlobalLM(globalLM);
		clrhdp.setMultiTheadFlag(false);
		clrhdp.loadUsers(analyzer.getUsers());
		clrhdp.setDisplayLv(displayLv);
		clrhdp.train();
		clrhdp.test();
	}
}
