package mains;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import Classifier.supervised.SVM;
import structures._Corpus;
import structures._Doc;
import clustering.KMeansAlg4Query;
import Analyzer.Analyzer;
import Analyzer.AspectAnalyzer;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.metricLearning.L2RWithQueryClustering;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.semisupervised.LCSReader;
import Classifier.semisupervised.LCSWriter;
import structures._Pair;
import topicmodels.LDA_Gibbs;
import topicmodels.pLSA;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import utils.Utils;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public class MyDocumentSelectionMain {
	
	
	public static void LCSWriteRead(_Corpus c){
		/************LCS Write and Read operations.***************/
		//Write the LCS to 8 files.
		int cores = Runtime.getRuntime().availableProcessors();
		Thread[] writeThreads = new Thread[cores];
		int LCSStart = 0, LCSEnd; int total = c.getCollection().size()-1;
		int LCSAvg = total / cores;
		for(int i=0; i < cores; i++){
			if(i == cores -1)
				LCSEnd = total;
			else LCSEnd = LCSStart + LCSAvg;
			writeThreads[i] = new Thread(new LCSWriter(LCSStart, LCSEnd, i, c.getCollection()));
			writeThreads[i].start();
			LCSStart = LCSEnd;
		}
		for(int i=0; i<writeThreads.length; i++){
			try {
				writeThreads[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		//Read the LCS from 8 files.
		HashMap<_Pair, Integer> LCSMap = new HashMap<_Pair, Integer>();
		Thread[] ReadThreads = new Thread[cores];
		LCSReader[] LCSReaders = new LCSReader[cores];
		for(int i=0; i < cores; i++){
			String filename = String.format("./data/LCS/LCS_%d", i);
			LCSReaders[i] = new LCSReader(filename);
			ReadThreads[i] = new Thread(LCSReaders[i]);
			ReadThreads[i].start();
		}
		for(int i=0; i<cores; i++){
			try {
				ReadThreads[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}	
		//Merge all the hashmaps to one single hashmap.
		for(int i=0; i<cores; i++){
			LCSMap.putAll(LCSReaders[i].getLCSMap());
		}
	}
	
//	public static void KmeansWrite(_Corpus c, int km){
//		//kmeans clutering among all review documents.
//		for (int k = 2; k <= 5; k++) {
//			KMeansAlg kmeans = new KMeansAlg(c, k);
//			kmeans.train(c.getCollection());
//			ArrayList<ArrayList<_Doc>> clusters = kmeans.getClustersDocs();
//
//			// Write out the results of clustering.
//			for (int i = 0; i < clusters.size(); i++) {
//				int[] stat = new int[5];
//				String filename = String.format("./data/kmeans/%dmeans_cluster_%d", k, i);
//				try{
//					PrintWriter printer = new PrintWriter(new File(filename));
//					for (_Doc d : clusters.get(i)) {
//						stat[d.getYLabel()]++;
//						printer.write(d.getYLabel() + "\n" + d.getSource() + "\n" + "******\n");
//					}
//					for (int j = 0; j < stat.length; j++)
//						printer.write(String.format("classNo: %d, count: %d, percentage: %.3f", j, stat[j], (double)j/(double)clusters.get(i).size()));
//					printer.close();
//				} catch(IOException e){
//					e.printStackTrace();
//				}
//			}
//		}
//	}
	
	//Print out randomly selected 100 reviews files.
	public static void print100Files(Analyzer analyzer, String fvStatFile){
		try{
			//Print out 100 documents to see the ratio.
			int count = 0, index = 0;
			ArrayList<_Doc> documents = analyzer.returnCorpus(fvStatFile).getCollection();
			_Doc[] selectedDocs = new _Doc[100];
			HashSet<Integer> checkIndexes = new HashSet<Integer>();
			PrintWriter writer = new PrintWriter(new File("./Selected100Files.txt"));
			Random r = new Random();
			while(count < 100){
				index = (int) (r.nextDouble()*documents.size());
				if(!checkIndexes.contains(index)){
					selectedDocs[count++] = documents.get(index);
					checkIndexes.add(index);
				}
			}
			for(int i=0; i<100; i++){
				writer.format("Label: %d\n", selectedDocs[i].getYLabel()+1);
				writer.write(selectedDocs[i].getSource() + "\n");
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM

		/*****parameters for the two-topic topic model*****/
//		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		int number_of_topics = 30;
//		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
//		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
//		int number_of_iteration = 100;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
//		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM"))
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model. Need it for postagging.
		String stopword = "./data/Model/stopwords.dat";
		String tagModel = "./data/Model/en-pos-maxent.bin";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		//Added by Mustafizur----------------
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String infoFilePath = "./data/result/"+"Topics_"+number_of_topics+"Information.txt";
		
		String category = "tablets"; //"electronics"
		String dataSize = "86jsons"; //"50K", "100K"
		
		/*****Parameters in learning style.*****/
		//"SUP", "SEMI"
		String style = "SEMI";
		
		//"RW", "RW-ML", "RW-L2R"
		String method = "RW-L2R-C";
				
		/*****Parameters in transductive learning.*****/
//		String debugOutput = String.format("data/debug/%s_topicmodel_diffProd.output", style);
		String debugOutput = String.format("data/debug/%s_%s_%s_%s_debug.output", style, method, category, dataSize);
		//k fold-cross validation
		int CVFold = 10; 
		//choice of base learner
		String multipleLearner = "SVM";
		//trade-off parameter
		double C = 1.0;
		
		/*****Parameters in feature selection.*****/
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.2; // Used in feature selection, the starting point of the features.
//		double endProb = 1.0; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
		
		String fvFile = String.format("./data/Features/fv_%dgram_%s_%s.txt", Ngram, category, dataSize);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, category, dataSize);
		
		int numOfAspects = 28; // 12, 14, 24, 28
		String aspectlist = String.format("./data/Model/%d_aspect_tablet.txt", numOfAspects);

//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
		Analyzer analyzer;
		_Corpus c;
		if(style.equals("SUP")){
			stnModel = null;
			analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.			
		} else{
			analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold, tagModel, aspectlist, true);
//			((DocAnalyzer) analyzer).setReleaseContent(false);
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			((DocAnalyzer) analyzer).LoadStopwords(stopword); //Load the sentiwordnet file.
			((DocAnalyzer) analyzer).loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//			analyzer.LoadTopicSentiment("./data/Sentiment/sentiment.csv", 2*number_of_topics);
			analyzer.setFeatureValues("TF", 0);		
//			c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

//			pLSA tModel = null;
//			if (topicmodel.equals("pLSA")) {			
//				tModel = new pLSA_multithread(number_of_iteration, converge, beta, c, 
//						lambda, number_of_topics, alpha);
//			} else if (topicmodel.equals("LDA_Gibbs")) {		
//				tModel = new LDA_Gibbs(number_of_iteration, converge, beta, c, 
//					lambda, number_of_topics, alpha, 0.4, 50);
//			}  else if (topicmodel.equals("LDA_Variational")) {		
//				tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
//						lambda, number_of_topics, alpha, 10, -1);
//			} else {
//				System.out.println("The selected topic model has not developed yet!");
//				return;
//			}
//		
//			tModel.setDisplay(true);
//			tModel.setInforWriter(infoFilePath);
//			tModel.setSentiAspectPrior(true);
//			tModel.LoadPrior(aspectlist, eta);
//			tModel.EMonCorpus();
//			//Write out the topK words under each topic.
//			int topK = 20;
//			String topWordsFile = String.format("./data/TopicVectors/%dAspects_top%dWords_corpus.txt", numOfAspects, topK);
//			tModel.writeTopWords(topWordsFile, topK);
		}
		
		String topicFile = String.format("./data/TopicVectors/%dAspects_topicVectors_corpus.txt", numOfAspects);
//		analyzer.saveTopicVectors(topicFile);
		analyzer.loadTopicVectors(topicFile, number_of_topics);
		
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

		//Write the average similarity into file.
		String SimFile = "./data/BoWTPJcdSimFile.txt";
//		((AspectAnalyzer) analyzer).calcAvgSimilarities();
//		((AspectAnalyzer) analyzer).writeAvgSimilarity(SimFile);
		
		//Load the average similarity into corpus.
		((AspectAnalyzer) analyzer).loadAvgSimilarity(SimFile);
		
		/***kmeans clustering***/
		int clusterNo = 5;
		int dim = 8; //Currently, we have 8 features for the query clustering.
		KMeansAlg4Query kmeans = new KMeansAlg4Query(c, clusterNo, dim);
		kmeans.train(c.getCollection());
		kmeans.setDocsClusterNo();
		
//		String kmeansStatFile = String.format("./data/kmeans/kmeans_stat_%d", noClusters);
//		String kmeansContentFile = String.format("./data/kmeans/kmeans_content_%d", noClusters);
//		kmeans.writeStat(kmeansStatFile);
//		kmeans.writeContent(kmeansContentFile);
		c.mapLabels(4); // Do kmeans first, then map the labels.
		
//		analyzer.LoadLCSFiles("./data/LCS");//Load LCS file from folder.
		
		if (style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1;
			int k = 20, kPrime = 20; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-4, tEta = 0.7; // convergence of random walk, weight of random walk
			boolean simFlag = false, weightedAvg = true;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			int topK = 20;
			double noiseRatio = 0, negRatio = 1; //0.5, 1, 1.5, 2
			int ranker = 0;//0-RankSVM; 1-lambda rank.
			
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
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg, 
						topK, noiseRatio, ranker, multithread_LR);
			} else if (method.equals("RW-L2R-C"))
				mySemi = new L2RWithQueryClustering(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg, 
						topK, noiseRatio, ranker, multithread_LR);
			
			mySemi.setKFold(CVFold);
			mySemi.setDebugOutput(debugOutput);
			((L2RWithQueryClustering) mySemi).setClusterNo(clusterNo);
//			((L2RMetricLearning) mySemi).setClusters(clusters);
//			((L2RMetricLearning) mySemi).setLCSMap(analyzer.returnLCSMap());
			mySemi.crossValidation(CVFold, c);

		} else if (style.equals("SUP")) {
			//perform supervised learning
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(c, C);
			mySVM.crossValidation(CVFold, c);
		}
	}
}
