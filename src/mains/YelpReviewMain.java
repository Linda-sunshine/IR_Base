package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import topicmodels.LDA_Gibbs;
import topicmodels.pLSA;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.Analyzer;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;

public class YelpReviewMain {
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 10; //Document length threshold
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "pLSA"; // pLSA, LDA_Gibbs, LDA_Variational
		
		int number_of_topics = 30;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-5, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int number_of_iteration = 100;
		
		/*****The parameters used in loading files.*****/
		String folder = "./YelpData/dedup";
		String suffix = ".json";
//		String folder = "./data/Electronics/dedup/RawData100K";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM"))
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
//		String stnModel = "./data/Model/en-sent.bin"; //Sentence model.
		String stopword = "./data/Model/stopwords.dat";
		String tagModel = "./data/Model/en-pos-maxent.bin";
				
		String category = "yelp"; //"electronics"
		String dataSize = "100"; //"50K", "100K"
		String fvFile = String.format("./YelpData/Features/fv_%dgram_%s_%s.txt", Ngram, category, dataSize);
		String fvStatFile = String.format("./YelpData/Features/fv_%dgram_stat_%s_%s.txt", Ngram, category, dataSize);
//		String aspectlist = "./data/Model/sentiment_output.txt";
//		String aspectlist = "./data/Model/topic_sentiment_output.txt";
		String aspectlist = "./data/Model/aspect_output_simple.txt";
		
		/*****Parameters in learning style.*****/
		//"SUP", "SEMI"
		String style = "SEMI";
		
		//"RW", "RW-ML", "RW-L2R"
		String method = "RW";
				
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
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.3; // Used in feature selection, the starting point of the features.
//		double endProb = 1.0; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
		Analyzer analyzer;
		_Corpus c;
		if(style.equals("SUP")){
			analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		} else{
			analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel);
//			analyzer = new AspectAnalyzer(tokenModel, null, classNumber, fvFile, Ngram, lengthThreshold, tagModel, aspectlist, true);
//			analyzer.setSentenceWriter("./data/input/BagOfSentencesLabels.txt");
			((DocAnalyzer) analyzer).LoadStopwords(stopword); //Load the sentiwordnet file.
//			((DocAnalyzer) analyzer).LoadSNWWithScore("./data/Model/SentiWordNet_3.0.0_20130122.txt");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//			analyzer.LoadTopicSentiment("./data/Sentiment/sentiment.csv", 2*number_of_topics);
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
		
			tModel.setDisplay(true);
//			tModel.LoadPrior(aspectlist, eta);
			tModel.EMonCorpus();
//			tModel.printTopWords(10, true);
		}
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		c.mapLabels(3);
		
		if (style.equals("SEMI")) {
			//perform transductive learning
			System.out.println("Start Transductive Learning, wait...");
			double learningRatio = 1;
			int k = 40, kPrime = 20; // k nearest labeled, k' nearest unlabeled
			double tAlpha = 1.0, tBeta = 0.1; // labeled data weight, unlabeled data weight
			double tDelta = 1e-4, tEta = 0.6; // convergence of random walk, weight of random walk
			boolean simFlag = false, weightedAvg = true;
			int bound = 0; // bound for generating rating constraints (must be zero in binary case)
			int topK = 6;
			double noiseRatio = 1.5, negRatio = 1; //0.5, 1, 1.5, 2
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
//			SVM mySVM = new SVM(c, C);
//			mySVM.crossValidation(CVFold, c);
		}
	}
}
