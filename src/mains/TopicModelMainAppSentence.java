package mains;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.ParseException;
import java.util.ArrayList;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures.Product;
import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LRHTMM;
import topicmodels.LRHTSM;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.LRHTSM_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.appReviewAnalyzer;
import Analyzer.newEggAnalyzer;

public class TopicModelMainAppSentence {

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Here it is the rating numbers. Since we are using sentence classification which is 0 and 1 is controlled separately
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 2; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM
		
		String category = "Games"; //Utilities, Travel, Sports, Social_Networking, Reference, Productivity, Photo_and_Video, News, Navigation, Music, Food_and_Drink, Finance, Education, Medical, Games, Business, Health_and_Fitness, Lifestyle
		/*****parameters for the topic models*****/
		double converge = 1e-9, lambda = 0.9; // negative converge means do need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5; // variational convergence
		int gibbs_iteration = 1500, gibbs_lag = 50;
		double burnIn = 0.4;
		boolean display = true;
		
		int crossV = 2; // crossV is 1 means all the data in trainset and anything greater than 1 means some testset
		boolean setRandomFold = true; // false means no shuffling and true means shuffling
		
		//String[] products = {"camera","tablet", "laptop", "phone", "surveillance", "tv"};
		// change topic number and category
		String topicmodel = "HTSM"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		int number_of_topics = 20;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		
		
		int loadAspectSentiPrior = 1; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from newEgg
		
    	int number_of_iteration = 20;
		int topK = 50;
		boolean sentence = false;
		boolean debugSentenceTransition = false;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/app/newdata/annotated/";
		String unfolder = "./data/app/newdata/un-annotated/";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		
		String featureDirectory = "./model/"+topicmodel+"/result/"+"app/";
		
		String fvFile = featureDirectory+"_topicmodel.txt"; 
		String fvStatFile = featureDirectory+"stat_topicmodel.txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";
		String pathToConjunctionWords = "./data/Model/conjunction.txt";
		String resultDirectory = "./model/"+topicmodel+"/result/app/";
		
		String aspectList = "./data/app/seed/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/app/seed/aspect_sentiment_"+ category + ".txt";
		
		
		
		String topWordFilePath = resultDirectory+"Topics_"+number_of_topics+"_topWords.txt";
		String wordIntrusionFilePath = resultDirectory+ topicmodel +"_" +category+"_Topics_" + number_of_topics + "_WordIntrusion.txt";
		String infoFilePath = resultDirectory+"Topics_"+number_of_topics+"_Information.txt";
		String featureWeightFilePath = resultDirectory + "Topics_"+number_of_topics+"_FeaturesWeight.csv";;
		String topicTransitionFilePath = resultDirectory + "Topics_"+number_of_topics+"_TopicTransitionMatrix.csv";;
		
		
		String debugDirectory = null;
		String debugFilePath = null;
		
		if(debugSentenceTransition){
			debugDirectory = "./model/"+topicmodel+"/debug/"+category+"/";
			debugFilePath = debugDirectory + "Topics_"+number_of_topics+"_Debug.csv";
			
		}
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.1; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 2; // Filter the features with DFs smaller than this threshold.
		
		/*System.out.println("Performing feature selection, wait...");
		appReviewAnalyzer analyzer = new appReviewAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.setCategory(category);
		analyzer.LoadStopwords(stopwords);
		//analyzer.readAnnotation(annotationFile);
		analyzer.LoadDirectory(folder, suffix, true);
		analyzer.LoadDirectory(unfolder, suffix, false);
		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		analyzer.printStat();*/
		
		
		
		System.out.println("Creating feature vectors, wait...");
		
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}
		
		
		appReviewAnalyzer analyzer = new appReviewAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
		analyzer.setCategory(category);
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords, pathToConjunctionWords);
		}
		
		//analyzer.readAnnotation(annotationFile);
		analyzer.LoadDirectory(folder, suffix, true);
		analyzer.LoadDirectory(unfolder, suffix, false);
			
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		ArrayList<String> featureSet = c.getAllFeatures();
	
		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, converge, beta, c, lambda, analyzer.getBackgroundProb());
			
			if (crossV<=1) {
				for(_Doc d:c.getCollection()) {
					model.inference(d);
					model.printTopWords(topK,topWordFilePath);
				}
			} else 
				model.crossValidation(crossV);
		} else if (topicmodel.equals("Tensor")) {
			c.saveAs3WayTensor("./data/vectors/3way_tensor.txt");
		} else {
			pLSA model = null;
			
			if (topicmodel.equals("pLSA")) {
				model = new pLSA_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {		
				model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, //in gibbs sampling, no need to compute log-likelihood during sampling
					lambda, number_of_topics, alpha, burnIn, gibbs_lag);
			}  else if (topicmodel.equals("LDA_Variational")) {		
				model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c, 
						lambda, number_of_topics, alpha, varIter, varConverge);
			} 
			else if (topicmodel.equals("HTMM")) {
				model = new HTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			} else if (topicmodel.equals("HTSM")) {
				model = new HTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha);
			}  
			else if (topicmodel.equals("LRHTMM")) {
				model = new LRHTMM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			}
			else if (topicmodel.equals("LRHTSM")) {
				model = new LRHTSM_multithread(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);
			}
			
			model.setDisplay(display);
			model.setInforWriter(infoFilePath);
			model.setNewEggLoadInTrain(loadNewEggInTrain);
			
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

			/*if(generateTrainTestDataForJSTASUM && setRandomFold==false){
				model.setFilePathForJSTASUM(trainSize,category, FilePath);
			}*/
			
			if(debugSentenceTransition){
				model.setDebugWriter(debugFilePath);
			}
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK, topWordFilePath);
			} else {
				model.setRandomFold(setRandomFold);
				model.crossValidation(crossV);
				model.printTopWords(topK, topWordFilePath);
				model.createWordIntrusionList(wordIntrusionFilePath);
				String topicWiseFEaturesFilePath =  resultDirectory + "Topics_" + number_of_topics + "_TopicWiseFeature.txt";
				if(topicmodel.equalsIgnoreCase("LRHTSM"))
					((HTMM)model).printTopFeaturesSet(topK, featureSet, topicWiseFEaturesFilePath);
			}
			
			if (sentence) {
				String summaryFilePath =  resultDirectory + topicmodel +"_" +category+"_Topics_" + number_of_topics + "_Summary.txt";
				model.setSummaryWriter(summaryFilePath);
//				if(category.equalsIgnoreCase("camera"))
//					((HTMM)model).docSummary(cameraProductList);
//				else if(category.equalsIgnoreCase("tablet"))
//					((HTMM)model).docSummary(tabletProductList);
//				else if(category.equalsIgnoreCase("phone"))
//					((HTMM)model).docSummary(phoneProductList);
//				else if(category.equalsIgnoreCase("tv"))
//					((HTMM)model).docSummary(tvProductList);
			}
			
			if(debugSentenceTransition && topicmodel.equalsIgnoreCase("LRHTSM")){
				model.debugOutputWrite();
			}
			if(topicmodel.equals("LRHTSM") && crossV>1){
				((LRHTSM)model).writeOmegaDelta(featureWeightFilePath);
				model.calculateTopicTransitionMatrix(topicTransitionFilePath);
			}
			
		}
	}
}
