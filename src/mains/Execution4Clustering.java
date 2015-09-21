package mains;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;

import clustering.KMeansAlg;
import structures.Parameter;
import structures._Corpus;
import structures._Doc;
import structures._Pair;
import topicmodels.LDA_Gibbs;
import topicmodels.pLSA;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.Analyzer;
import Analyzer.AspectAnalyzer;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.semisupervised.LCSReader;
import Classifier.supervised.SVM;
/**
 * @author Lin
 */
public class Execution4Clustering  {
	static public void main(String[] args) throws IOException, ParseException {
		Parameter param = new Parameter(args);
		System.out.println(param.toString());
		
		//Used in experiments. added by Lin.
		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int number_of_iteration = 100;
		
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String infoFilePath = "./data/result/"+"Topics_"+number_of_topics+"Information.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";
		String aspectlist = "./data/Model/aspect_output_simple.txt";
		String tagModel = "./data/Model/en-pos-maxent.bin";
		
		int CVFold = 10; 
		double C = 1.0;
		String method = "RW-L2R";
		String multipleLearner = "SVM";
		String debugOutput = String.format("data/debug/%s_topK%d_noCluster%d_debug.output", method, param.m_topK, param.m_noC);

		//String stnModel = (param.m_model.equals("HTMM")||param.m_model.equals("LRHTMM"))?param.m_stnModel:null;
		//String posModel = (param.m_model.equals("HTMM")||param.m_model.equals("LRHTMM"))?param.m_posModel:null;
		
		System.out.println("Creating feature vectors, wait...");
		Analyzer analyzer;
		_Corpus c;
		if(param.m_style.equals("SUP")){
			param.m_stnModel = null;
			analyzer = new jsonAnalyzer(param.m_tokenModel, param.m_classNumber, param.m_fvFile, param.m_Ngram, param.m_lengthThreshold, param.m_stnModel);
			analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
		} else{
			analyzer = new AspectAnalyzer(param.m_tokenModel, param.m_stnModel, param.m_classNumber, param.m_fvFile, param.m_Ngram, param.m_lengthThreshold, tagModel, aspectlist, true);
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			((DocAnalyzer) analyzer).LoadStopwords(param.m_stopwords); //Load the sentiwordnet file.
			((DocAnalyzer) analyzer).loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		
			analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
//			analyzer.LoadTopicSentiment("./data/Sentiment/sentiment.csv", 2*number_of_topics);
			analyzer.setFeatureValues("TF", 0);		
			c = analyzer.returnCorpus(param.m_featureStat); // Get the collection of all the documents.

			pLSA tModel = null;
			if (topicmodel.equals("pLSA")) {			
				tModel = new pLSA_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {		
				tModel = new LDA_Gibbs(number_of_iteration, converge, beta, c, 
					lambda, number_of_topics, alpha, 0.4, 50);
			}  else if (topicmodel.equals("LDA_Variational")) {		
				tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha, 10, -1);
			} else {
				System.out.println("The selected topic model has not developed yet!");
				return;
			}
		
			tModel.setDisplay(true);
			tModel.setInforWriter(infoFilePath);
			tModel.setSentiAspectPrior(true);
			tModel.LoadPrior(aspectlist, eta);
			tModel.EMonCorpus();	

		}
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(param.m_featureFile); // Get the collection of all the documents.
//		c.mapLabels(4);
		
		
		/************LCS Write and Read operations.***************/
//		int cores = Runtime.getRuntime().availableProcessors();
		//Write the LCS to 8 files.
//		Thread[] writeThreads = new Thread[cores];
//		int LCSStart = 0, LCSEnd; int total = c.getCollection().size()-1;
//		int LCSAvg = total / cores;
//		for(int i=0; i < cores; i++){
//			if(i == cores -1)
//				LCSEnd = total;
//			else LCSEnd = LCSStart + LCSAvg;
//			writeThreads[i] = new Thread(new LCSWriter(LCSStart, LCSEnd, i, c.getCollection()));
//			writeThreads[i].start();
//			LCSStart = LCSEnd;
//		}
//		for(int i=0; i<writeThreads.length; i++){
//			try {
//				writeThreads[i].join();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
//		}
		
//		//Read the LCS from 8 files.
//		HashMap<_Pair, Integer> LCSMap = new HashMap<_Pair, Integer>();
//		Thread[] ReadThreads = new Thread[cores];
//		LCSReader[] LCSReaders = new LCSReader[cores];
//		for(int i=0; i < cores; i++){
//			String filename = String.format("./data/LCS/LCS_%d", i);
//			LCSReaders[i] = new LCSReader(filename);
//			ReadThreads[i] = new Thread(LCSReaders[i]);
//			ReadThreads[i].start();
//		}
//		for(int i=0; i<cores; i++){
//			try {
//				ReadThreads[i].join();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
//		}	
//		//Merge all the hashmaps to one single hashmap.
//		for(int i=0; i<cores; i++){
//			LCSMap.putAll(LCSReaders[i].getLCSMap());
//		}
//		
//		//kmeans clutering among all review documents.
//		// Do clustering first.
//		KMeansAlg kmeans = new KMeansAlg(c, param.m_noC);
//		kmeans.train(c.getCollection());
//		ArrayList<ArrayList<_Doc>> clusters = kmeans.getClusters();
				
//		KMeansAlg kmeans = new KMeansAlg(c, param.m_noC);
//		kmeans.train(c.getCollection());
//		kmeans.tranferClusters2Docs();
//		ArrayList<ArrayList<_Doc>> clusters = kmeans.getClustersDocs();
//		
//		String kmeansStatFile = String.format("./data/kmeans/kmeans_stat_%d", param.m_noC);
//		String kmeansContentFile = String.format("./data/kmeans/kmeans_content_%d", param.m_noC);
//
//		kmeans.writeStat(kmeansStatFile);
//		kmeans.writeContent(kmeansContentFile);
		
		c.mapLabels(4); //Do kmeans first, then map the labels.
		
		if (param.m_style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1;
			int k = 20, kPrime = 20; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-4, tEta = 0.8; // convergence of random walk, weight of random walk
			boolean simFlag = true, weightedAvg = true;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			//int topK = 5;
			double noiseRatio = 1.5, negRatio = 1; //0.5, 1, 1.5, 2
			int ranker = 1;//0-RankSVM; 1-lambda rank.
			boolean metricLearning = true;
			boolean multithread_LR = true;//training LambdaRank with multi-threads

			GaussianFieldsByRandomWalk mySemi = null;			
			if (method.equals("RW")) {
				mySemi = new GaussianFieldsByRandomWalk(c, multipleLearner, C,
					learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg); 
			} else if (method.equals("RW-ML")) {
				mySemi = new LinearSVMMetricLearning(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, false, 
						bound);
				((LinearSVMMetricLearning)mySemi).setMetricLearningMethod(metricLearning);
			} else if (method.equals("RW-L2R")) {
				mySemi = new L2RMetricLearning(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, param.m_eta, weightedAvg, 
						param.m_topK, param.m_noiseRatio, ranker, multithread_LR);
			}
//			((L2RMetricLearning) mySemi).setQueryRatio(param.m_queryRatio);
//			((L2RMetricLearning) mySemi).setDocumentRatio(param.m_documentRatio);
			((L2RMetricLearning) mySemi).setLambda(param.m_lambda4L2R);
			((L2RMetricLearning) mySemi).setShrinkage(param.m_shrinkage);
			((L2RMetricLearning) mySemi).setStepSize(param.m_stepSize);
			((L2RMetricLearning) mySemi).setWindowSize(param.m_windowSize);
			((L2RMetricLearning) mySemi).setMaxIter(param.m_maxIter);
			
			mySemi.setSimilarity(simFlag);
			mySemi.setDebugOutput(debugOutput);
//			((L2RMetricLearning) mySemi).setClusters(clusters);
//			((L2RMetricLearning) mySemi).setLCSMap(LCSMap);
			System.out.print(String.format("------Parameters--------\n" +
					"Learning ratio: %.3f\tKUL:%d\tKUU:%d\talpha:%.3f\tbeta:%.3f\n" +
					"eta:%.3f\ttopK:%d\tnoiseRatio:%.3f\tranker:%d\n" +
					"lambda:%.3f\tshrinkage:%.3f\tstepSize:%.3f\twindowsize:%d\tmaxIteration:%d\n", 
					learningRatio, k, kPrime, tAlpha, tBeta, param.m_eta, param.m_topK, param.m_noiseRatio, ranker, 
					param.m_lambda4L2R, param.m_shrinkage, param.m_stepSize, param.m_windowSize, param.m_maxIter));
			mySemi.crossValidation(CVFold, c);
		} else if (param.m_style.equals("SUP")) {
			//perform supervised learning
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(c, C);
			mySVM.crossValidation(CVFold, c);
		}
	}
}

