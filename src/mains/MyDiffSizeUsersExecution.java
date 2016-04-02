package mains;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import structures.MTLinAdaptParameter;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;

public class MyDiffSizeUsersExecution {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws IOException{
		
		MTLinAdaptParameter param = new MTLinAdaptParameter(args);

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", param.m_data); // CV.
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", param.m_data);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", param.m_data);
//		String folder = String.format("/if15/lg5bt/DataSigir/%s/", param.m_data);
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", param.m_data); // CV.
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", param.m_data);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", param.m_data);
		String folder = String.format("/home/lin/DiffSetsUsers/");
		
		MultiThreadedUserAnalyzer analyzer;
		String diffFolder, filename;
		// We need ten sets of experiments to do the average.
		double[][] F1 = new double[param.m_ttlSizeSet][];
		int size = 0;
		for(int i=0; i< param.m_ttlSizeSet; i++){
			size = 400 + 400*i;
			diffFolder = String.format("%sUsers_%d/Users_%d", folder, param.m_userSet, size);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(diffFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
			 
			if(param.m_model.equals("mtsvm")){
				//Create the instance of MT-SVM
				MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
				mtsvm.setPersonlized(false);
				mtsvm.loadUsers(analyzer.getUsers());
				mtsvm.setBias(true);
				mtsvm.train();
				mtsvm.test();
				F1[i]= mtsvm.getPerf();
			}
			else if(param.m_model.equals("mtlinadapt")){
				// Create instance of MTLinAdaptWithSupUsr
				MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
				mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
				mtlinadaptsup.loadUsers(analyzer.getUsers());
				mtlinadaptsup.setDisplayLv(displayLv);
				mtlinadaptsup.setR1TradeOffs(param.m_eta1, param.m_eta2);
				mtlinadaptsup.setRsTradeOffs(param.m_lambda1, param.m_lambda2);
			
				mtlinadaptsup.train();
				mtlinadaptsup.test();
				F1[i] = mtlinadaptsup.getPerf();
			}
		}
		filename = String.format("/if15/lg5bt/DataSigir/%s_%s_%.1f_Users_%d.txt", param.m_data, param.m_model, param.m_userSet, adaptRatio, param.m_userSet);
		PrintWriter writer = new PrintWriter(new File(filename));
		for(int i=0; i<F1.length; i++)
			writer.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\n", (i+1)*400, F1[i][0], F1[i][1], F1[i][2], F1[i][3]);
		writer.close();
	}
}

