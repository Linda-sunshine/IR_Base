package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncGeneral;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.asyncMTLinAdapt;

public class MySplitMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int topKNeighbors = 20;
		int displayLv = 2;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		// Best performance for mt-linadapt in amazon.
//		double eta1 = 1, eta2 = 0.5, lambda1 = 0.1, lambda2 = 0.3;
		// Best performance for mt-linadapt in yelp.
		double eta1 = 0.9, eta2 =1 , lambda1 = 0.1, lambda2 = 0.1;
		boolean enforceAdapt = true;

		String dataset = "Yelp"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
			
		String trainSize = "1000";
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users_split_1", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
				
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);
//		String featureGroupFile = String.format("/if15/lg5bt/DataSigir/%s/CrossGroups_800.txt", dataset);
//		String globalModel = String.format("/if15/lg5bt/DataSigir/%s/GlobalWeights.txt", dataset);
		
		String model = "mtlinadapt";
		boolean train = true, test = false;
		String supModel = String.format("./SuperModel/%s_%s_super.txt", model, trainSize);
		
		MultiThreadedUserAnalyzer analyzer;
		if(train){
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.setReleaseContent(true);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(userFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

			if(model.equals("mtlinadapt")){
				//Create an instance of MTLinAdapt with Super user sharing different dimensions.
				MTLinAdapt mtlinadaptsup = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile); 
				mtlinadaptsup.loadFeatureGroupMap4SupUsr(null);//featureGroupFileSup
				mtlinadaptsup.loadUsers(analyzer.getUsers());
				mtlinadaptsup.setDisplayLv(displayLv);
				mtlinadaptsup.setR1TradeOffs(eta1, eta2);
				mtlinadaptsup.setRsTradeOffs(lambda1, lambda2);
				mtlinadaptsup.train();
				mtlinadaptsup.saveSupModel(supModel);
				mtlinadaptsup.test();
			} else if(model.equals("mtsvm")){		
				//Create the instance of MT-SVM
				MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
				mtsvm.loadUsers(analyzer.getUsers());
				mtsvm.setBias(false);
				mtsvm.train();
				mtsvm.saveSupModel(supModel);
				mtsvm.test();		
			} 
		}
//		GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
//		gsvm.loadUsers(analyzer.getUsers());
//		gsvm.setBias(true);
//		gsvm.train();
//		gsvm.test();
//		gsvm.saveSupModel("globalsvm.txt");
			
		if(test){
			userFolder = String.format("./data/CoLinAdapt/%s/Users_split_2", dataset);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.setReleaseContent(true);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(userFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			HashMap<String, Integer> featureMap = analyzer.getFeatureMap();

			// Create an instances of asyncMTLinAdapt model.
			asyncMTLinAdapt adaptation = new asyncMTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, null);
			adaptation.loadGlobal(String.format("./SuperModel/mtlinadapt_%s_super.txt", trainSize));
			adaptation.loadUsers(analyzer.getUsers());
			adaptation.setDataset(dataset);
			adaptation.setTrainByUser(false);
			adaptation.setDisplayLv(displayLv);
			adaptation.train();
			adaptation.test();	
		
			userFolder = String.format("./data/CoLinAdapt/%s/Users_split_2", dataset);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.setReleaseContent(true);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(userFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			featureMap = analyzer.getFeatureMap();

			//Create the instance of MT-SVM
			asyncGeneral general = new asyncGeneral("mtsvm", classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
			general.loadUsers(analyzer.getUsers());
			general.setDataset(dataset);
			general.loadGlobal(String.format("./SuperModel/mtsvm_%s_super.txt", trainSize));
			general.train();
			general.test();	
		
			userFolder = String.format("./data/CoLinAdapt/%s/Users_split_2", dataset);
			analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
			analyzer.setReleaseContent(true);
			analyzer.config(trainRatio, adaptRatio, enforceAdapt);
			analyzer.loadUserDir(userFolder); // load user and reviews
			analyzer.setFeatureValues("TFIDF-sublinear", 0);
			featureMap = analyzer.getFeatureMap();

			//Create the instance of Global model.
			general = new asyncGeneral("global", classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
			general.loadUsers(analyzer.getUsers());
			general.setDataset(dataset);
			general.train();
			general.test();	
		}
	}
}
