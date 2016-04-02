package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import opennlp.tools.util.InvalidFormatException;

public class ComparisonWithDiffUsersMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.1;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
//		double eta1 = 0.5, eta2 = 1;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
//		String folder = String.format("/if15/lg5bt/DataSigir/%s/", dataset);
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/home/lin/DiffSetsUsers/");
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		
		MultiThreadedUserAnalyzer analyzer;
		//Access users of different size.
		int size = 0;
		String diffFolder, filename;
		// We need ten sets of experiments to do the average.
		for(int t=0; t<10; t++){
		int i = 0;
			double[][] F1 = new double[25][4];
//			for(int i=0; i<F1.length; i++){
				size = 400 + 400 * i;
				diffFolder = String.format("%sUsers_%d/Users_%d", userFolder, t+1, size);
				analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
				analyzer.config(trainRatio, adaptRatio, enforceAdapt);
				analyzer.loadUserDir(diffFolder); // load user and reviews
				analyzer.setFeatureValues("TFIDF-sublinear", 0);
				HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
				
				//Create the instance of MT-SVM
				MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
				mtsvm.setPersonlized(false);
				mtsvm.loadUsers(analyzer.getUsers());
				mtsvm.setBias(true);
				mtsvm.train();
				mtsvm.test();
				F1[i] = mtsvm.getPerf();
				
				// Create instance of MTLinAdaptWithSupUsr
				MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
				mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
				mtlinadaptsup.loadUsers(analyzer.getUsers());
				mtlinadaptsup.setDisplayLv(displayLv);
				mtlinadaptsup.setR1TradeOffs(eta1, eta2);
				mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
				
				mtlinadaptsup.train();
				mtlinadaptsup.test();
				F1[i] = mtlinadaptsup.getPerf();
			
				System.out.format("%d users in testing.\n", size);
				for(double f: F1[i])
					System.out.format("%.4f\t", f);
				System.out.println();
//			}
			filename = String.format("/if15/lg5bt/DataSigir/%sPerformance/MTSVM_u_1_0.5/MTSVM_%s_0.5_Users_%d.txt", dataset, dataset, t+1);
//			filename = String.format("/if15/lg5bt/DataSigir/%sPerformance/MTLinAdaptSup_0.5/MTLinAdaptSup_0.5_Users_%d.txt", dataset, t+1);
//			filename = String.format("/if15/lg5bt/DataSigir/%sPerformance/YelpComparison/MTLinAdaptSup_Yelp_0.5_Users_%d.txt", dataset, t+1);

//			PrintWriter writer = new PrintWriter(new File(filename));
//			for(int i=0; i<F1.length; i++)
//				writer.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\n", (i+1)*400, F1[i][0], F1[i][1], F1[i][2], F1[i][3]);
//			writer.close();
		}
	}
}
