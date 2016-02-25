package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTCoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdaptWithSupUsr;

public class MyMTLinAdaptMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		double eta1 = 0.5, eta2 = 1, eta3 = 0.50, eta4 = 1, neighborsHistoryWeight = 0.5;
//		double eta1 = 50, eta2 = 100, eta3 = 0.5, eta4 = 1, neighborsHistoryWeight = 0.5;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
//		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
			
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
		
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

		// Create an instances of CoLinAdapt model.
//		CoLinAdapt adaptation = new CoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//				
//		adaptation.loadUsers(analyzer.getUsers());
//		adaptation.setDisplayLv(displayLv);
//		//adaptation.setTestMode(TestMode.TM_batch);
//		adaptation.setR1TradeOffs(eta1, eta2);
//		adaptation.setR2TradeOffs(eta3, eta4);
//		adaptation.train();
//		adaptation.test();
//		//adaptation.saveModel("data/results/colinadapt");
		
		//Create the instance of MTLinAdapt.
//		double lambda1 = 1, lambda2 = 0.5;
//		MTLinAdapt mtlinadapt = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
//		mtlinadapt.loadUsers(analyzer.getUsers());
//		mtlinadapt.setDisplayLv(displayLv);
//		mtlinadapt.setR1TradeOffs(eta1, eta2);
//		mtlinadapt.setR2TradeOffs(eta3, eta4);
//		mtlinadapt.setRsTradeOffs(lambda1, lambda2);
//		
//		mtlinadapt.train();
//		mtlinadapt.printWeights("./data/MTLinAdapt/weights");
//		mtlinadapt.test();
		double[] lambda1s = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		double[] lambda2s = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
		for(double lambda1: lambda1s){
			for(double lambda2: lambda2s){
		
//		double lambda1 = 0.5, lambda2 = 1;
		MTLinAdaptWithSupUsr mtlinadaptsup = new MTLinAdaptWithSupUsr(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
		mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
		mtlinadaptsup.loadUsers(analyzer.getUsers());

		mtlinadaptsup.setDisplayLv(displayLv);
		mtlinadaptsup.setR1TradeOffs(eta1, eta2);
		mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
		
		mtlinadaptsup.train();
		mtlinadaptsup.test();
		}}
		
		// Create the instance of MTCoLinAdapt.
//		MTCoLinAdapt mtcolinadapt = new MTCoLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile);
//		mtcolinadapt.loadUsers(analyzer.getUsers());
//		mtcolinadapt.setDisplayLv(displayLv);
//		mtcolinadapt.setR1TradeOffs(eta1, eta2);
//		mtcolinadapt.setR2TradeOffs(eta3, eta4);
//		mtlinadapt.setRsTradeOffs(lambda1, lambda2);
//
//		mtcolinadapt.train();
//		mtcolinadapt.test();
//		_AdaptStruct u;
//		ArrayList<_AdaptStruct> mtlinUsr = adaptation.getUserList();
//		double[][] mtlinStat = new double[mtlinUsr.size()][3]; //[0]: F1; [1]: number of reviews; [2]: polarity:
//		for(int i=0; i<mtlinUsr.size(); i++){
//			u = mtlinUsr.get(i);
//			mtlinStat[i][0] = u.getUser().getPerfStat().getF1(0) + u.getUser().getPerfStat().getF1(1);
//			mtlinStat[i][1] = u.getUser().getReviewSize();
//			mtlinStat[i][2] = u.getUser().calculatePosRatio();
//			u.getUser().getPerfStat().clear();
//		}
		
		//Create the instance of MT-SVM
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.setBias(true);
//		mtsvm.train();
//		mtsvm.test();
	
//		ArrayList<_AdaptStruct> mtsvmUsr = mtsvm.getUserList();
//		double[][] mtsvmStat = new double[mtsvmUsr.size()][2];
//		for(int i=0; i<mtsvmUsr.size(); i++){
//			u = mtsvmUsr.get(i);
//			mtsvmStat[i][0] = u.getUser().getPerfStat().getF1(0) + u.getUser().getPerfStat().getF1(1);
//			mtsvmStat[i][1] = u.getUser().getReviewSize();
//		}
//		PrintWriter writer0 = new PrintWriter(new File("./data/MTLinAdapt/Equal.txt"));
//		PrintWriter writer1 = new PrintWriter(new File("./data/MTLinAdapt/MtlinAdaptBetter.txt"));
//		PrintWriter writer2 = new PrintWriter(new File("./data/MTLinAdapt/MtsvmBetter.txt"));
//
//		// Get stat of the two set of users.
//		double[] stat = new double[6];
//		ArrayList<Double> equal = new ArrayList<Double>();
//		ArrayList<Double> mtlinAdaptBetter = new ArrayList<Double>();
//		ArrayList<Double> mtsvmBetter = new ArrayList<Double>();
//		
//		for(int i=0; i<mtlinStat.length; i++){
//			
//			if(mtlinStat[i][0] == mtsvmStat[i][0]){
//				stat[0]++;
//				stat[3] += mtlinStat[i][1];
//				equal.add(mtlinStat[i][1]);
//				writer0.write(mtsvmStat[i][1]+"\t" + mtlinStat[i][2] + "\n");
//			} else if(mtlinStat[i][0] > mtsvmStat[i][0]){
//				stat[1]++;
//				stat[4] += mtlinStat[i][1];
//				mtlinAdaptBetter.add(mtlinStat[i][1]);
//				writer1.format("%.4f,%.4f,%.1f,%.4f\n", mtlinStat[i][0], mtsvmStat[i][0], mtlinStat[i][1], mtlinStat[i][2]);
//			} else{
//				stat[2]++;
//				stat[5] += mtsvmStat[i][1];
//				mtsvmBetter.add(mtlinStat[i][1]);
//				writer2.format("%.4f,%.4f,%.1f,%.4f\n", mtlinStat[i][0], mtsvmStat[i][0], mtsvmStat[i][1], mtlinStat[i][2]);
//			}
//		}
//		writer0.close();
//		writer1.close();
//		writer2.close();
//		
//		double[] groupStat = getStat(equal);
//		System.out.println("Equal group info:");
//		System.out.format("light: %.4f\tmedium: %.4f\theavy: %.4f\n", groupStat[0], groupStat[1], groupStat[2]);
//		
//		groupStat = getStat(mtlinAdaptBetter);
//		System.out.println("MTLinAdapt better group info:");
//		System.out.format("light: %.4f\tmedium: %.4f\theavy: %.4f\n", groupStat[0], groupStat[1], groupStat[2]);
//		
//		groupStat = getStat(mtsvmBetter);
//		System.out.println("MTSVM better group info:");
//		System.out.format("light: %.4f\tmedium: %.4f\theavy: %.4f\n", groupStat[0], groupStat[1], groupStat[2]);
//		
//		stat[3] /= stat[0];
//		stat[4] /= stat[1];
//		stat[5] /= stat[2];
//		
//		System.out.println("mtlin==mtsvm\tmtlin>mtsvm\tmtlin<mtsvm\tavgequal\tavgmtlin\tavgmtsvm");
//		System.out.format("%.1f\t%.1f\t%.1f\t%.4f\t%.4f\t%.4f\n", stat[0], stat[1], stat[2], stat[3], stat[4], stat[5]);
	}
	
	public static double[] getStat(ArrayList<Double> list){
		double[] count = new double[3];
		for(int i=0; i<list.size(); i++){
			if(list.get(i) <= 10)
				count[0]++;
			else if(list.get(i) <=50)
				count[1]++;
			else 
				count[2]++;
		}
		for(int i=0; i<count.length; i++)
			count[i] /= list.size();
		return count;
	}
}
