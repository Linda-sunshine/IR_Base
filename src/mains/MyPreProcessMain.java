package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._User;
import Analyzer.Analyzer;
import Analyzer.CrossFeatureSelection;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.NewUserAnalyzer;
import Analyzer.UserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.SVM;
import Classifier.supervised.modelAdaptation.HDP.MTCLRWithHDP;

public class MyPreProcessMain {
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
	
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String providedCV = null;
		String dataset = "AmazonNew"; // "Amazon", "AmazonNew", "Yelp"

		int trainSize = 3; // "3"
		int userSize = 9; // "20"
		String trainDir = String.format("./data/%s/Users_%dk", dataset, trainSize);
		String userDir = String.format("./data/%s/Users_%dk",dataset, userSize);

//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String featureGroupFileB = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);

		/**Feature selection**/
		double startProb = 0; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
		int topK = 7000; // topK for language model.

		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		String pattern = String.format("%dgram_%s", Ngram, featureSelection);
		String fvFile = String.format("./data/%s/fv_%dk_%s.txt", dataset, trainSize, pattern);
		String fvFile4LM = String.format("./data/%s/fv_%dk_lm_%d_%s.txt", dataset, trainSize, topK, pattern);
		String globalModel = String.format("./data/%s/GlobalWeights_%dk.txt", dataset, trainSize);
		
		// Multi-thread may need access to some global variables at the same time. 
//		UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold);
		
//		// Feature selection for language model.
//		analyzer.LoadStopwords(stopwords);
//		analyzer.loadUserDir(userFolder);
//		analyzer.featureSelection(fvFile, featureSelection, maxDF, minDF, topK);
//		
//		// Feature selection for logistic model.
//		featureSelection = "CHI";
//		analyzer.LoadStopwords(stopwords);
//		analyzer.loadUserDir(userFolder);
//		analyzer.featureSelection(fvFile4LM, featureSelection, startProb, endProb, maxDF, minDF);		// Save global model weights.
		
		/**Train Global model**/
		NewUserAnalyzer analyzer = new NewUserAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.loadUserDir(trainDir);
		SVM svm = new SVM(classNumber, analyzer.getFeatureSize(), 1);
//		svm.train(analyzer.mergeReviews());
		svm.saveModel(globalModel);
		
		/**Cross feature groups**/
		int kFold = 10, kmeans = 800;
		String crossfv = String.format("./data/%s/CrossFeatures_%dk_%d_%d/", dataset, trainSize, kFold, kmeans);
		ArrayList<_Doc> crossDocs = (ArrayList<_Doc>) analyzer.mergeReviews();
		CrossFeatureSelection crossfs = new CrossFeatureSelection(crossDocs, classNumber, analyzer.getFeatureSize(), kFold, kmeans);
		crossfs.train();
		crossfs.kMeans(crossfv);
	}
}
