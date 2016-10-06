package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import clustering.KMeansAlg4Vct;
import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVMWithClusters;

public class MyMTSVMClusterSVMWeightsMain {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.25;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users_1000", dataset);
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
			
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		analyzer.constructSparseVector4Users(); 

		// Train individual svms and save weights.
		String indsvmFolder = "./data/models/individualsvm_0.25/", suffix = "classifer";
		IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
		indsvm.loadUsers(analyzer.getUsers());
		indsvm.train();
		indsvm.saveModel(indsvmFolder);
//		//Load the weights learned from the individual svms.
		analyzer.loadUserWeights(indsvmFolder, suffix); 
		
		// We perform kmeans over user weights learned from individual svms.
		int[] kmeans = new int[]{10, 50};
		for(int kmean: kmeans){
		KMeansAlg4Vct alg = new KMeansAlg4Vct(analyzer.getUsers(), kmean, analyzer.getFeatureSize());
		alg.train();
		
		// The returned clusters contain the corresponding cluster index of each user.
		int[] clusters = alg.getClusters();
		
		// Take one cluster as one power user with all user members' reviews inside.
//		Collection<_User> rawUserGroups = analyzer.groupUsers(clusters).values();
//		IndividualSVM svm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
//		ArrayList<_User> userGroups = new ArrayList<_User>();
//		userGroups.addAll(rawUserGroups);
//		svm.loadUsers(userGroups);
//		svm.train();
//		svm.test();
//		System.out.print(String.format("------------------Individual SVM finishes here.----------------\n"));

		double[] cs = new double[]{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0};
		for(double c: cs){
		
		// Create an instance of mtsvm with clusters.
		MultiTaskSVMWithClusters mtsvmcluster = new MultiTaskSVMWithClusters(classNumber, analyzer.getFeatureSize(), kmean, clusters);
		mtsvmcluster.loadUsers(analyzer.getUsers());
		mtsvmcluster.setAllParams(1, c, 1);
		mtsvmcluster.train();
		mtsvmcluster.test();

		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		}
		
		// Create an intance of mtsvm.
//		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
//		mtsvm.loadUsers(analyzer.getUsers());
//		mtsvm.train();
//		mtsvm.test();
		}
	}
}
