package mains;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

import structures.DPParameter;
import structures.MTLinAdaptParameter;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.DirichletProcess.MTCLinAdaptWithDP;
import Classifier.supervised.modelAdaptation.RegLR.RegLR4Sup;

public class MyDiffSizeUsersExecution {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws IOException{
		
		DPParameter param = new DPParameter(args);

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int topKNeighbors = 20;
		int displayLv = 0;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", param.m_data); // CV.
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", param.m_data);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", param.m_data);
		String folder = String.format("/if15/lg5bt/DataSigir/%s/", param.m_data);
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", param.m_data); // CV.
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", param.m_data);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", param.m_data);
//		String folder = String.format("/home/lin/DiffSetsUsers/");
		
		MultiThreadedUserAnalyzer analyzer;
		String diffFolder, filename;
		// We need ten sets of experiments to do the average.
		for(int j=0; j<param.m_ttlUserSetNo; j++){
			double[][] F1 = new double[param.m_ttlSizeSet][];
			int size = 0;
			for(int i=0; i< param.m_ttlSizeSet; i++){
				size = 400 + 400*i;
				diffFolder = String.format("%sUsers_%d/Users_%d", folder, j+1, size);
				analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
				analyzer.config(trainRatio, adaptRatio, enforceAdapt);
				analyzer.loadUserDir(diffFolder); // load user and reviews
				analyzer.setFeatureValues("TFIDF-sublinear", 0);
				HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
			 
				if(param.m_model.equals("mtsvm")){
					//Create the instance of MT-SVM
					MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
					mtsvm.loadUsers(analyzer.getUsers());
					mtsvm.setBias(true);
					mtsvm.train();
					mtsvm.test();
					F1[i]= mtsvm.getPerf();
				}
				else if(param.m_model.equals("mtclindp")){
					MTCLinAdaptWithDP mtclindp = new MTCLinAdaptWithDP(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, null);
					mtclindp.loadUsers(analyzer.getUsers());
					mtclindp.setDisplayLv(displayLv);
					mtclindp.setLNormFlag(false);
					mtclindp.setsdA(0.4);
					mtclindp.setsdB(0.02);
					mtclindp.setR1TradeOffs(0.01, 0.01);
					mtclindp.setR2TradeOffs(0.03, 0.03);
					mtclindp.train();
					mtclindp.test();
					mtclindp.printInfo();
					F1[i] = mtclindp.getPerf();
				}
			}
			filename = String.format("/if15/lg5bt/DataWsdm/%s_%s_%.1f_Users_%d.txt", param.m_data, param.m_model, adaptRatio, j+1);
			PrintWriter writer = new PrintWriter(new File(filename));
			for(int i=0; i<F1.length; i++)
				writer.format("%d\t%.4f\t%.4f\t%.4f\t%.4f\n", (i+1)*400, F1[i][0], F1[i][1], F1[i][2], F1[i][3]);
			writer.close();
		}
	}
}

