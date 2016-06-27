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
		
		String[] opts = new String[]{"G", "D", "G+D"};
		for(String i: opts){
			
		System.out.print(String.format("-------------------%s----------------\n", i));
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setDFScheme(i);
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
		int[] kmeans = new int[]{10, 50, 100, 200, 400, 800, 1600};
		for(int kmean: kmeans){
		KMeansAlg4Vct alg = new KMeansAlg4Vct(analyzer.getUsers(), kmean, analyzer.getFeatureSize());
		alg.train();
		
		// The returned clusters contain the corresponding cluster index of each user.
		int[] clusters = alg.getClusters();
		
		// Take one cluster as one power user with all user members' reviews inside.
		Collection<_User> rawUserGroups = analyzer.groupUsers(clusters).values();
		IndividualSVM svm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
		ArrayList<_User> userGroups = new ArrayList<_User>();
		userGroups.addAll(rawUserGroups);
		svm.loadUsers(userGroups);
		svm.train();
		svm.test();
		System.out.print(String.format("------------------Individual SVM finishes here.----------------\n"));

		// Create an instance of mtsvm with clusters.
		MultiTaskSVMWithClusters mtsvmcluster = new MultiTaskSVMWithClusters(classNumber, analyzer.getFeatureSize(), kmean, clusters);
		mtsvmcluster.loadUsers(analyzer.getUsers());
		mtsvmcluster.train();
		mtsvmcluster.test();
		System.out.print(String.format("------------------MultiTaskSVM with clusters finishes here.----------------\n"));

		for(_User u: analyzer.getUsers())
			u.getPerfStat().clear();
		
		// Create an intance of mtsvm.
		MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
		mtsvm.loadUsers(analyzer.getUsers());
		mtsvm.train();
		mtsvm.test();
		System.out.print(String.format("------------------MultiTaskSVM finishes here.----------------\n"));
		}}
	}
}
