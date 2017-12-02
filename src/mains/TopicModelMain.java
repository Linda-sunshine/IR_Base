package mains;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

import Analyzer.ParentChildAnalyzer;
import structures._Corpus;
import structures._Doc;
import topicmodels.twoTopic;
import topicmodels.DCM.DCMLDA_test;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.LDA.LDA_Gibbs_test;
import topicmodels.correspondenceModels.ACCTM;
import topicmodels.correspondenceModels.ACCTM_C;
import topicmodels.correspondenceModels.ACCTM_CHard;
import topicmodels.correspondenceModels.ACCTM_CZ;
import topicmodels.correspondenceModels.ACCTM_CZLR;
import topicmodels.correspondenceModels.DCMCorrLDA_multi_E_test;
import topicmodels.correspondenceModels.corrLDA_Gibbs;
import topicmodels.markovmodel.HTMM;
import topicmodels.markovmodel.HTSM;
import topicmodels.markovmodel.LRHTMM;
import topicmodels.markovmodel.LRHTSM;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;

public class TopicModelMain {

	public static void main(String[] args) throws IOException, ParseException {	
		
		int mb = 1024*1024;
		
		Runtime rTime = Runtime.getRuntime();
		System.out.println("totalMem\t:"+rTime.totalMemory()/mb);
		
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each document should have at least 2 sentences
		
		/*****parameters for the two-topic topic model*****/
		//ACCTM, ACCTM_TwoTheta, ACCTM_C, ACCTM_CZ, ACCTM_CZLR, LDAonArticles, ACCTM_C, 
		// correspondence_LDA_Gibbs, LDA_Gibbs_Debug, LDA_Variational_multithread
		// 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM,
		
		String topicmodel = "DCMCorrLDA_multi_E_test";


		String category = "tablet";
		int number_of_topics = 30;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = true; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 20, number_of_iteration = 50, crossV = 1;

		int gibbs_iteration = 1000, gibbs_lag = 50;
		int displayLap = 20;
		// gibbs_iteration = 4;
		// gibbs_lag = 2;
		// displayLap = 2;
		
		double burnIn = 0.4;

		boolean sentence = false;
		
		// most popular items under each category from Amazon
		// needed for docSummary
		String tabletProductList[] = {"B008GFRDL0"};
		String cameraProductList[] = {"B005IHAIMA"};
		String phoneProductList[] = {"B00COYOAYW"};
		String tvProductList[] = {"B0074FGLUM"};
		
		/*****The parameters used in loading files.*****/
		String amazonFolder = "./data/amazon/tablet/topicmodel";
		String newEggFolder = "./data/NewEgg";
		String articleType = "Tech";
//		articleType = "Gadgets";
//		 articleType = "Yahoo";
//		articleType = "APP";
		
		String articleFolder = String.format(
				"./data/ParentChildTopicModel/%sArticles",
						articleType);
		String commentFolder = String.format(
				"./data/ParentChildTopicModel/%sComments",
						articleType);

		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || 
				topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM")) {
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}

		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram, articleType);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, articleType, topicmodel);
		
		String aspectList = "./data/Model/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		File rootFolder = new File("./data/results");
		if(!rootFolder.exists()){
			System.out.println("creating root directory"+rootFolder);
			rootFolder.mkdir();
		}
		
		SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyyMMdd-HHmm");	
		String filePrefix = String.format("./data/results/%s", dateFormatter.format(new Date()));
		filePrefix = filePrefix + "-" + topicmodel + "-" + articleType;
		File resultFolder = new File(filePrefix);
		if (!resultFolder.exists()) {
			System.out.println("creating directory" + resultFolder);
			resultFolder.mkdir();
		}
		
		String outputFile = filePrefix + "/consoleOutput.txt";
		PrintStream printStream = new PrintStream(new FileOutputStream(
				outputFile));
		System.setOut(printStream);
		
		String infoFilePath = filePrefix + "/Information.txt";
		////store top k words distribution over topic
		String topWordPath = filePrefix + "/topWords.txt";
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 5; // Filter the features with DFs smaller than this threshold.

		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);

//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.		

		/***** parent child topic model *****/
		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(commentFolder, suffix);
		if(topicmodel.equals("LDA_APPMerged"))
			articleFolder = String.format(
					"./data/ParentChildTopicModel/%sDescriptionsReviews",
					articleType);	
//		articleFolder = String.format(
//				"./data/ParentChildTopicModel/%sArticles4Merged",
//				articleType);
//		
//		commentFolder = String.format(
//				"./data/ParentChildTopicModel/%sComments4Merged",
//				articleType);
//		
		analyzer.LoadParentDirectory(articleFolder, suffix);
		// analyzer.LoadDirectory(articleFolder, suffix);
		// analyzer.LoadDirectory(commentFolder, suffix);

		analyzer.LoadChildDirectory(commentFolder, suffix);

//		if((topicmodel."LDA_APP")&&(topicmodel!="LDA_APPMerged"))
//		analyzer.LoadChildDirectory(commentFolder, suffix);
		
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
		
//		analyzer.LoadNewEggDirectory(newEggFolder, suffix); //Load all the documents as the data set.
//		analyzer.LoadDirectory(amazonFolder, suffix);			
		
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.	
//		_Corpus c = analyzer.getCorpus();
		
		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, converge, beta, c, lambda);
			
			if (crossV<=1) {
				for(_Doc d:c.getCollection()) {
					model.inference(d);
					model.printTopWords(topK);
				}
			} else 
				model.crossValidation(crossV);
		} else {
			pLSA model = null;
			
			if (topicmodel.equals("pLSA")) {
				model = new pLSA_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {
				number_of_topics = 15;
				model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, //in gibbs sampling, no need to compute log-likelihood during sampling
					lambda, number_of_topics, alpha, burnIn, gibbs_lag);
			} else if (topicmodel.equals("LDA_Variational_multithread")) {		
				model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha, varIter, varConverge);
			} else if (topicmodel.equals("HTMM")) {
				model = new HTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("HTSM")) {
				model = new HTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("LRHTMM")) {
				model = new LRHTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			} else if (topicmodel.equals("LRHTSM")) {
				model = new LRHTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			} else if(topicmodel.equals("correspondence_LDA_Gibbs")){
				double ksi = 800;
				double tau = 0.7;
				model = new corrLDA_Gibbs(gibbs_iteration, 0, beta-1, c, //in gibbs sampling, no need to compute log-likelihood during sampling
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag);
			}else if(topicmodel.equals("ACCTM")){
				double mu = 1.0;
				double[] gamma = {0.5, 0.5};
				double ksi = 800;
				double tau = 0.7;
				model = new ACCTM(gibbs_iteration, 0, beta-1, c,
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag);
			}else if(topicmodel.equals("ACCTM_C")){
				double mu = 1.0;
				double[] gamma = {0.5, 0.5};
				beta = 1.001;
				double ksi = 800;
				double tau = 0.7;
				converge = 1e-5;
				model = new ACCTM_C(gibbs_iteration, 0, beta-1, c,
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag,
						gamma);
			}else if(topicmodel.equals("ACCTM_CHard")){
				double mu = 1.0;
				double[] gamma = {0.5, 0.5};
				double ksi = 800;
				double tau = 0.7;
				beta = 1.001;
				model = new ACCTM_CHard(gibbs_iteration, 0, beta-1, c,
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag,
						gamma);
			}else if(topicmodel.equals("ACCTM_CZ")){
				double mu = 1.0;
				double[] gamma = {0.5, 0.5};
				beta = 1.001;
				alpha = 1.01;
				double ksi = 800;
				double tau = 0.7;
				number_of_topics = 30;
				model = new ACCTM_CZ(gibbs_iteration, 0, beta-1, c,
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag,
						gamma);
			}else if(topicmodel.equals("ACCTM_CZLR")){
				double mu = 1.0;
				double[] gamma = {0.5, 0.5};
				beta = 1.001;
				alpha = 1.01;
				double ksi = 800;
				double tau = 0.7;
				number_of_topics = 30;
				converge = 1e-9;
				model = new ACCTM_CZLR(gibbs_iteration, converge, beta-1, c, 
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag, gamma);
			} else if (topicmodel.equals("DCMLDA_test")) {
				converge = 1e-3;
				int newtonIter = 50;
				double newtonConverge = 1e-3;
				number_of_topics = 15;
				model = new DCMLDA_test(gibbs_iteration, converge, beta - 1, c,
						lambda, number_of_topics, alpha - 1, burnIn, gibbs_lag,
						newtonIter, newtonConverge);
			} else if (topicmodel.equals("LDA_Gibbs_test")) {
				number_of_topics = 15;
				// in gibbs sampling, no need to compute
				// log-likelihood during sampling
				model = new LDA_Gibbs_test(gibbs_iteration, 0, beta, c,
						lambda, number_of_topics, alpha, burnIn, gibbs_lag);
			} else if (topicmodel.equals("DCMCorrLDA_multi_E_test")) {
				number_of_topics = 15;
				converge = 1e-3;
				int newtonIter = 50;
				double newtonConverge = 1e-3;
				number_of_topics = 15;
				double ksi = 800;
				double tau = 0.7;
				double alphaC = 0.001;
				model = new DCMCorrLDA_multi_E_test(gibbs_iteration, converge,
						beta - 1, c, lambda, number_of_topics, alpha - 1,
						alphaC,
						burnIn, ksi, tau, gibbs_lag,
						newtonIter, newtonConverge);
			}
			
			model.setDisplayLap(displayLap);
			model.setInforWriter(infoFilePath);
//			model.setNewEggLoadInTrain(loadNewEggInTrain);
			
			if(loadAspectSentiPrior==1){
				System.out.println("Loading aspect-senti list from "+aspectSentiList);
				model.setSentiAspectPrior(true);
				model.LoadPrior(aspectSentiList, eta);
			} else if(loadAspectSentiPrior==2){
				System.out.println("Loading aspect list from "+aspectList);
				model.setSentiAspectPrior(false);
				model.LoadPrior(aspectList, eta);
			}else{
				System.out.println("No prior is added!!");
			}
						
			if (crossV<=1) {
				model.EMonCorpus();
				if(topWordPath == null)
					model.printTopWords(topK);
				else
					model.printTopWords(topK, topWordPath);
			} else {
				model.setRandomFold(setRandomFold);
				double trainProportion = 0.5;
				double testProportion = 1-trainProportion;
				model.setPerplexityProportion(testProportion);
				model.crossValidation(crossV);
				model.printTopWords(topK, topWordPath);
			}
			
			model.closeWriter();
			
			if (sentence) {
				String summaryFilePath =  "./data/results/Topics_" + number_of_topics + "_Summary.txt";
				model.setSummaryWriter(summaryFilePath);
				if(category.equalsIgnoreCase("camera"))
					((HTMM)model).docSummary(cameraProductList);
				else if(category.equalsIgnoreCase("tablet"))
					((HTMM)model).docSummary(tabletProductList);
				else if(category.equalsIgnoreCase("phone"))
					((HTMM)model).docSummary(phoneProductList);
				else if(category.equalsIgnoreCase("tv"))
					((HTMM)model).docSummary(tvProductList);
			}
		}
	}
}
