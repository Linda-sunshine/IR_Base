package mains;

import java.io.IOException;
import java.text.ParseException;

import Analyzer.AspectAnalyzer;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;

public class TransductiveMain {
	
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM

		/*****parameters for the two-topic topic model*****/
		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int number_of_iteration = 100;
		boolean aspectSentiPrior = true;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablet/topicmodel";
		String suffix = ".json";
		String stopword = "./data/Model/stopwords.dat";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model. Need it for pos tagging.		
		String tagModel = "./data/Model/en-pos-maxent.bin";
		String sentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		//Added by Mustafizur----------------
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		
//		String category = "tablets"; //"electronics"
//		String dataSize = "86jsons"; //"50K", "100K"
//		String fvFile = String.format("./data/Features/fv_%dgram_%s_%s.txt", Ngram, category, dataSize);
//		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, category, dataSize);
//		String aspectlist = "./data/Model/aspect_output_simple.txt";
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
		String aspectSentiList = "./data/Model/aspect_sentiment_tablet.txt";
		String aspectList = "./data/Model/aspect_tablet.txt";

		/*****Parameters in learning style.*****/
		//"SUP", "SEMI"
		String style = "SEMI";
		
		//"RW", "RW-ML", "RW-L2R"
		String method = "RW-L2R";
				
		/*****Parameters in transductive learning.*****/
		String debugOutput = "data/debug/topical.sim";
//		String debugOutput = null;
		boolean releaseContent = false;
		//k fold-cross validation
		int CVFold = 10; 
		//choice of base learner
		String multipleLearner = "SVM";
		//trade-off parameter
		double C = 1.0;
		
		/*****Parameters in feature selection.*****/
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.5; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		System.out.println("Creating feature vectors, wait...");
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, tagModel, classNumber, fvFile, Ngram, lengthThreshold, aspectList, true);
		//Added by Mustafizur----------------
		analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
		analyzer.LoadStopwords(stopword); // Load the sentiwordnet file.
		analyzer.loadPriorPosNegWords(sentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		analyzer.setReleaseContent(releaseContent);
		
		// Added by Mustafizur----------------
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
		analyzer.setFeatureValues("TF", 0);		
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

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
		
		tModel.setDisplayLap(0);
		tModel.setSentiAspectPrior(aspectSentiPrior);
		tModel.LoadPrior(aspectSentiPrior?aspectSentiList:aspectList, eta);
		tModel.EMonCorpus();	
		
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c.mapLabels(3);//how to set this reasonably
		
		if (style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1.0;
			int k = 30, kPrime = 20; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 0.1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-5, tEta = 0.6; // convergence of random walk, weight of random walk
			boolean simFlag = false, weightedAvg = true;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			int topK = 25; // top K similar documents for constructing pairwise ranking targets
			double noiseRatio = 1.0;
			boolean metricLearning = true;
			boolean multithread_LR = true;//training LambdaRank with multi-threads
			
			GaussianFieldsByRandomWalk mySemi = null;			
			if (method.equals("RW")) {
				mySemi = new GaussianFieldsByRandomWalk(c, multipleLearner, C,
					learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg); 
			} else if (method.equals("RW-ML")) {
				mySemi = new LinearSVMMetricLearning(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg, 
						bound);
				((LinearSVMMetricLearning)mySemi).setMetricLearningMethod(metricLearning);
			} else if (method.equals("RW-L2R")) {
				mySemi = new L2RMetricLearning(c, multipleLearner, C, 
						learningRatio, k, kPrime, tAlpha, tBeta, tDelta, tEta, weightedAvg, 
						topK, noiseRatio, multithread_LR);
			}
			
			mySemi.setSimilarity(simFlag);
			mySemi.setDebugOutput(debugOutput);
			mySemi.crossValidation(CVFold, c);
		} else if (style.equals("SUP")) {
			//perform supervised learning
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(c, C);
			mySVM.crossValidation(CVFold, c);
		}
	}

}
