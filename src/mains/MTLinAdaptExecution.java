package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
import structures.MTLinAdaptParameter;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.GlobalSVM;
import Classifier.supervised.IndividualSVM;
import Classifier.supervised.modelAdaptation.Base;
import Classifier.supervised.modelAdaptation.MultiTaskSVM;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt.MTLinAdapt;
import Classifier.supervised.modelAdaptation.RegLR.RegLR;

public class MTLinAdaptExecution {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		MTLinAdaptParameter param = new MTLinAdaptParameter(args);
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = param.m_adaptRatio;
		int topKNeighbors = 200;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;
		String tokenModel = "./data/Model/en-token.bin"; // Token model.

		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", param.m_data); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", param.m_data);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", param.m_data, param.m_fvGroup);
		String featureGroupFileSup = String.format("./data/CoLinAdapt/%s/CrossGroups_%d.txt", param.m_data, param.m_fvGroupSup);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", param.m_data);
		
		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
		HashMap<String, Integer> featureMap = analyzer.getFeatureMap();
		
		// If we have 5000 features, then just pass null in.
		if(param.m_fvGroup == 5000)
			featureGroupFile = null;
		if(param.m_fvGroupSup == 5000)
			featureGroupFileSup = null;
		
		if(param.m_model.equals("base")){
			Base base  = new Base(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
			base.loadUsers(analyzer.getUsers());
			base.setPersonalizedModel();
			base.test();
		}else if(param.m_model.equals("gsvm")){
			GlobalSVM gsvm = new GlobalSVM(classNumber, analyzer.getFeatureSize());
			gsvm.loadUsers(analyzer.getUsers());
			gsvm.setBias(false);
			gsvm.train();
			gsvm.test();
		}else if(param.m_model.equals("indsvm")){
			IndividualSVM indsvm = new IndividualSVM(classNumber, analyzer.getFeatureSize());
			indsvm.loadUsers(analyzer.getUsers());
			indsvm.setBias(false);
			indsvm.train();
			indsvm.test();
		}else if(param.m_model.equals("reglr")){
			RegLR reglr = new RegLR(classNumber, analyzer.getFeatureSize(), featureMap, globalModel);
			reglr.loadUsers(analyzer.getUsers());
			reglr.setDisplayLv(displayLv);
			reglr.train();
			reglr.test();
		}else if(param.m_model.equals("linadapt")){
			LinAdapt linadapt = new LinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile);
			linadapt.loadUsers(analyzer.getUsers());
			linadapt.setDisplayLv(displayLv);
			linadapt.train();
			linadapt.test();
		}else if(param.m_model.equals("mtsvm")){
			MultiTaskSVM mtsvm = new MultiTaskSVM(classNumber, analyzer.getFeatureSize());
			mtsvm.loadUsers(analyzer.getUsers());
			mtsvm.setBias(true);
			mtsvm.train();
			mtsvm.test();
		}else if(param.m_model.equals("mtlinbatch")){
			MTLinAdapt mtlinadapt = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, featureGroupFileSup);
			mtlinadapt.loadUsers(analyzer.getUsers());
			mtlinadapt.setDisplayLv(displayLv);
			mtlinadapt.setR1TradeOffs(param.m_eta1, param.m_eta2);
			mtlinadapt.setR2TradeOffs(param.m_eta3, param.m_eta4);
			mtlinadapt.train();
			mtlinadapt.test();
		}else if(param.m_model.equals("mtlinonline")){
			MTLinAdapt mtlinadapt = new MTLinAdapt(classNumber, analyzer.getFeatureSize(), featureMap, topKNeighbors, globalModel, featureGroupFile, featureGroupFileSup);
			mtlinadapt.loadUsers(analyzer.getUsers());
			mtlinadapt.setDisplayLv(displayLv);
			mtlinadapt.setR1TradeOffs(param.m_eta1, param.m_eta2);
			mtlinadapt.setR2TradeOffs(param.m_eta3, param.m_eta4);
			mtlinadapt.train();
			mtlinadapt.test();
		}
	}
}
