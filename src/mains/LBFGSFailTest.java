package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithSupUsr;
import opennlp.tools.util.InvalidFormatException;

public class LBFGSFailTest {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 10, eta2 = 10, lambda1 = 10, lambda2 = 10;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		String folder = String.format("/home/lin/DiffSetsUsers/");
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String folder = String.format("/if15/lg5bt/DataSigir/%s/", dataset);
		
		MultiThreadedUserAnalyzer analyzer;
		//Access users of different size.
		int size = 0;
		String diffFolder;
		
		for(int t=0; t<10; t++){
			size = 4000;
			diffFolder = String.format("%sUsers_%d/Users_%d", folder, t+1,size);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(diffFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
				
			// Create instance of MTLinAdaptWithSupUsr
			MTLinAdaptWithSupUsr mtlinadaptsup = new MTLinAdaptWithSupUsr(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
			mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
			mtlinadaptsup.loadUsers(analyzer.getUsers());
			mtlinadaptsup.setLNormFlag(false);
			mtlinadaptsup.setDisplayLv(displayLv);
			mtlinadaptsup.setR1TradeOffs(eta1, eta2);
			mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
			mtlinadaptsup.train();
			mtlinadaptsup.test();
		}
//		// We need ten sets of experiments to do the average.
//		double[] ps = new double[]{0.6, 0.7, 0.8, 0.9, 1};
//		double[] qs = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
//		double[][] flags = new double[24][10];
//		for(double lambda1: ps){
//			for(double lambda2: qs){
//				for(int t=0; t<10; t++){
//					for(int i=0; i<24; i++){
//						size = 400 + 400 * i;
//						diffFolder = String.format("%sUsers_%d/Users_%d", folder, t+1, size);
//						analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
//						analyzer.config(trainRatio, adaptRatio, enforceAdapt);
//						analyzer.loadUserDir(diffFolder); // load user and reviews
//						analyzer.setFeatureValues("TFIDF-sublinear", 0);
//						HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
//				
//						// Create instance of MTLinAdaptWithSupUsr
//						MTLinAdaptWithSupUsr mtlinadaptsup = new MTLinAdaptWithSupUsr(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
//						mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
//						mtlinadaptsup.loadUsers(analyzer.getUsers());
//						mtlinadaptsup.setDisplayLv(displayLv);
//						mtlinadaptsup.setR1TradeOffs(eta1, eta2);
//						mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
//						mtlinadaptsup.train();
//						flags[i][t] = mtlinadaptsup.getLBFGSFlag();
//					}
//				}
//				String filename = String.format("%.1f_%.1f_%.1f_%.1f.txt", eta1, eta2, lambda1, lambda2);
//				PrintWriter writer = new PrintWriter(new File(filename));
//				for(double[] flag: flags){
//					for(double f: flag)
//						writer.write(f+"\t");
//					writer.write("\n");
//				}
//				writer.close();
//			}
//		}
	}
}
