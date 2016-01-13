package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.UserAnalyzer;
import CoLinAdapt.CoLinAdaptSchedule;
import CoLinAdapt.MultiTaskSVM;

public class CoLinAdaptMain {

	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		
		int classNumber = 2;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int featureGroupNo = 400; //There should be some feature grouping methods.
		
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String providedCV = "./data/CoLinAdapt/5000Features.txt"; // CV.// CV.
		String CVStat = "./data/CoLinAdapt/SelectedVocab.csv";
		String featureGroupFile = "./data/CoLinAdapt/CrossGroups.txt";
		
		String globalModel = "./data/CoLinAdapt/GlobalWeights.txt";
		String folder = "./data/CoLinAdapt/Users2";
		
		String svdFile = "./data/CoLinAdapt/SVD/UserWordMatrix_nonN_U.mm";
		
//		UserAnalyzer analyzer;
//		//Access users of different size.
//		int size = 0;
//		String diffFolder;
//		double[][] F1 = new double[2][50];
//		for(int i=1; i<51; i++){
//			size = 200 * i;
//			diffFolder = folder + "Users_" + size;
//			analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
//			analyzer.LoadCVStat(CVStat);
//			analyzer.loadUserDir(diffFolder);//Load users.	
//			analyzer.setFeatureValues(25265, "TFIDF", 0);
//			analyzer.loadGlobalWeights(globalModel);
//			analyzer.loadFeatureGroupIndexes(featureGroupFile);
//			
//			//Multi-task SVM training and testing.
//			MultiTaskSVM MTSVM = new MultiTaskSVM(analyzer.getUsers(), analyzer.getFeatureSize());
//			MTSVM.batchTrainTest();
//			MTSVM.calcPerformance();
//			
//			System.out.format("%d users in testing.\n", size);
//			MTSVM.printPerformance();
//			F1[0][i-1] = MTSVM.getNegF1();
//			F1[1][i-1] = MTSVM.getPosF1();
//		}
//		
//		PrintWriter writer = new PrintWriter(new File("./data/MT_SVM_Performance.txt"));
//		for(int i=0; i<F1[0].length; i++)
//			writer.format("%d\t%.4f\t%.4f\n", (i+1)*200, F1[0][i], F1[1][i]);
//		writer.close();
		
		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		analyzer.LoadCVStat(CVStat);
		analyzer.loadUserDir(folder);//Load users.	
		analyzer.setFeatureValues(25265, "TFIDF", 0);
		analyzer.loadGlobalWeights(globalModel);
		analyzer.loadFeatureGroupIndexes(featureGroupFile);
//		analyzer.loadSVD(svdFile);// Load low dimension representation of users.
	
		
		//Create the instances of the CoLinAdapt model.
		CoLinAdaptSchedule coLinAdaptS = new CoLinAdaptSchedule(analyzer.getUsers(), analyzer.getFeatureSize(), featureGroupNo, analyzer.getFeatureGroupIndexes());
		coLinAdaptS.setGlobalWeights(analyzer.getGlobalWeights());
		
		int topK = 5;
		double eta1 = 0.5, eta2 = 0.01, eta3 = 0.8, eta4 = 0.1;
		coLinAdaptS.setCoefficients4R1(eta1, eta2);
		coLinAdaptS.setCoefficients4R2(eta3, eta4);
		
		coLinAdaptS.calcluateSimilarities("cosine");//"Euc", "cosine"
		
		coLinAdaptS.initSchedule();
		String neighborFile = "./data/CoLinAdapt/Neighbor";
		coLinAdaptS.constructNeighborhood(topK);

//		coLinAdaptS.writeUserNeighbors(neighborFile, topK);
//		coLinAdaptS.loadUserNeighbors(neighborFile, topK);
		
		//Online training.
//		System.out.println("Start online training....");
//		coLinAdaptS.onlineTrain();
//		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
//		coLinAdaptS.printPerformance();
		
		//Batch training.
		System.out.println("Start batch training....");
		coLinAdaptS.initSchedule();
		coLinAdaptS.batchTrainTest();
		coLinAdaptS.calcPerformance(); //Calculate the performance of each user.	
		coLinAdaptS.printPerformance();
	}
}
