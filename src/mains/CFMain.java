package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;

import Analyzer.MultiThreadedUserAnalyzer;
import Application.CollaborativeFiltering;

public class CFMain {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 5;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
//		double eta1 = 0.5, eta2 = 0.5, eta3 = 0.6, eta4 = 0.01, neighborsHistoryWeight = 0.5;
		double eta1 = 1.3087, eta2 = 0.0251, eta3 = 1.7739, eta4 = 0.4859, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = false;

		String dataset = "Yelp"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String folder = String.format("/if15/lg5bt/DataSigir/%s/", dataset);
		
			
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
//		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
	
//		String svdFile = "./data/CoLinAdapt/Amazon/Amazon_SVD.mm";
//		analyzer.loadSVDFile(svdFile);
				
		/***Collaborative filtering starts here.***/
//		for(int t=0; t<10; t++){
		// Construct the rdmNeighbors first.
		String model, dir, weightFile="";
		int topK, time = 5;
		CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), time);
		cfInit.init();
		HashMap<String, ArrayList<Integer>> userIDRdmNeighbors = cfInit.constructRandomNeighbors();

		String[] models = new String[]{"colinadapt", "CoRegLR", "MTSVM", "BoW", "linadapt"};
		//"linadapt", "CoRegLR", "MTSVM",
//		int[] ks = new int[]{4, 6};
		int[] ks = new int[]{2, 4};
		String suffix = "classifer";
		double[][] performance = new double[models.length][ks.length*2];
		for(int m=0; m<models.length; m++){
			model = models[m];
			weightFile = dataset + "_1_10";
			dir = String.format("/home/lin/DataSigir2016/classifiers/%s/%s", weightFile, model);//"coLinAdapt", "linAdapt", "MTSVM"
//			dir = String.format("/if15/lg5bt/DataSigir/classifiers/%s/%s", weightFile, model);//"coLinAdapt", "linAdapt", "MTSVM"
			for(int n=0; n < ks.length; n++){
				topK = ks[n];
				System.out.format("-----------------run %s %d neighbors-------------------------\n", model, topK);
				CollaborativeFiltering cf = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, topK, time, model);
				cf.setUserIDRdmNeighbors(userIDRdmNeighbors);
				cf.init();
				cf.loadWeights(dir, suffix);
				cf.calculatAllNDCGMAP();
				String singleFile = String.format("./data/%s_%d.txt", model, topK);
				cf.calcuateSaveAvgNDCGMAP(singleFile);
				performance[m][2*n] =  cf.getAvgNDCG();
				performance[m][2*n + 1] = cf.getAvgMAP();
				double avgNuUsers = cf.getItemStat();
				System.out.println("The average number of users for each item is " + avgNuUsers);
			}
		}
		String filename = String.format("./data/%s_test_cf_performance_1.txt", weightFile);
//		String filename = String.format("/if15/lg5bt/DataSigir/CF/%s_test_cf_performance_%d.txt", dataset, t);
		PrintWriter writer = new PrintWriter(new File(filename));
		
		writer.write("\t\t");
		for(int k: ks)
			writer.write("NDCG\tMAP\t");
		writer.write("\n");

		for(int m=0; m<models.length; m++){
			writer.write(models[m]+"\t");
			for(double p: performance[m]){
				writer.write(p+"\t");
			}
			writer.write("\n");
		}
		writer.close();
	}
//	}
}
