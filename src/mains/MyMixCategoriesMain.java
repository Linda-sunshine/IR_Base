package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
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
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;

import Classifier.supervised.SVM;

public class MyMixCategoriesMain {
	
	public static void main(String[] args) throws IOException, ParseException {	
				
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 10; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM

		/*****parameters for the two-topic topic model*****/
		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.6; // negative converge means do need to check likelihood convergency
		int number_of_iteration = 100;
		
		/*****The parameters used in loading files.*****/
		String category = "13"; //"electronics"
		String dataSize = "500"; //"50K", "100K"
		String folder = String.format("./data/MixCategories/%s_%sCategories", category, dataSize);
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
		
		String fvFile = String.format("./data/Features/fv_%dgram_mixed_%s_%s.txt", Ngram, category, dataSize);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_mixed_%s_%s.txt", Ngram, category, dataSize);
		String aspectlist = "./data/Model/aspect_output_simple.txt";

		/*****Parameters in learning style.*****/
		//"SUP", "SEMI"
		String style = "SEMI";
		
		//"RW", "RW-ML", "RW-L2R"
		String method = "RW-L2R";
				
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
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "IG"; //Feature selection method.
//		double startProb = 0.2; // Used in feature selection, the starting point of the features.
//		double endProb = 1.0; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 20; // Filter the features with DFs smaller than this threshold.
//		System.out.println("Performing feature selection, wait...");
//		DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadSNAPFiles(folder);
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
//		DocAnalyzer analyzer = new DocAnalyzer(tokenModel, stnModel, tagModel, classNumber, fvFile, Ngram, lengthThreshold);
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold, tagModel);
		analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
		analyzer.LoadStopwords(stopword); //Load the sentiwordnet file.
		analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		analyzer.LoadSNAPFiles(folder); //Load all the documents as the data set.
		_Corpus c;
		if(style.equals("SEMI")){
			analyzer.setFeatureValues("TF", 0);		
			c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.

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
			tModel.setSentiAspectPrior(true);
			tModel.LoadPrior(aspectlist, eta);
			tModel.EMonCorpus();	

		}
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		c.mapLabels(4);
		
		if (style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1;
			int k = 20, kPrime = 20; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-4, tEta = 0.7; // convergence of random walk, weight of random walk
			boolean simFlag = false, weightedAvg = true;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			int topK = 6;
			double noiseRatio = 1.5, negRatio = 1; //0.5, 1, 1.5, 2
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
			}
//			mySemi.setSimiMethod("T");
			mySemi.setDebugOutput(debugOutput);
			mySemi.crossValidation(CVFold, c);
//			mySemi.printInfo();
		} else if (style.equals("SUP")) {
			//perform supervised learning
			System.out.println("Start SVM, wait...");
			SVM mySVM = new SVM(c, C);
			mySVM.crossValidation(CVFold, c);
		}
	}
}
