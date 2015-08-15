package mains;

import java.io.IOException;
import java.text.ParseException;
import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LRHTMM;
import topicmodels.LRHTSM;
import topicmodels.pLSA;
import topicmodels.pLSAGroup;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.LRHTSM_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.jsonAnalyzer;
import Analyzer.newEggAnalyzer;

public class TopicModelMainnew {

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM
		
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "LRHTSM"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM
		//String[] products = {"camera","tablet", "laptop", "phone", "surveillance", "tv"};
		
		
		// change topic number and category
		int crossV = 2; // crossV is 1 means all the data in trainset and anything greater than 1 means some testset
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int testDocMod = 11; // when setRandomFold = false, we select every m_testDocMod_th document for testing

		String category = "camera";
		int number_of_topics = 26;
		int loadAspectSentiPrior = 2; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		boolean loadNewEggInTrain = false; // false means in training there is no reviews from newEgg
		int loadProsCons = 2; // 0 means only load pros, 1 means load only cons, 2 means load both pros and cons 
		int trainSize = 1000; // trainSize 0 means only newEgg; 5000 means all the data 
		int number_of_iteration = 50;
		int topK = 50;
		boolean generateTrainTestDataForJSTASUM = false;
		boolean sentence = false;
		boolean debugSentenceTransition = false;
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		double converge = 1e-9, lambda = 0.9; // negative converge means do need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int gibbs_iteration = 1500, gibbs_lag = 50;
		double burnIn = 0.4;
		boolean display = true;
		
		// most popular items under each category from Amazon
		String tabletProductList[] = {"B008DWG5HE"};
		String cameraProductList[] = {"B005IHAIMA"};
		String phoneProductList[] = {"B00COYOAYW"};
		String tvProductList[] = {"B0074FGLUM"};
		
		/*****The parameters used in loading files.*****/
		//String folder = "./data/amazon/tablet/topicmodel";
		
		String amazonFolder = "./data/amazon/mainData/"+ category + "/"+trainSize+"/";
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = null;
		String posModel = null;
		
		
		String featureDirectory = "./data/amazon/mainData/"+category+"/"+trainSize+"/";
		if(loadNewEggInTrain)
			featureDirectory += "NewEggLoaded/";
		else
			featureDirectory += "noNewEgg/";
		if(loadAspectSentiPrior==0)
			featureDirectory += "noPrior/";
		else if(loadAspectSentiPrior==1)
			featureDirectory += "aspectSentiPrior/";
		else
			featureDirectory += "aspectPrior/";
		
		String fvFile = featureDirectory+"_topicmodel.txt"; 
		String fvStatFile = featureDirectory+"stat_topicmodel.txt";
		
		String aspectList = "./data/Model/aspect_"+ category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";
		
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		String FilePath = "./data/amazon/";
		String resultDirectory = "./result/"+category+"/"+trainSize+"/";
		if(loadNewEggInTrain)
			resultDirectory += "NewEggLoaded/";
		else
			resultDirectory += "noNewEgg/";
		if(loadAspectSentiPrior==0)
			resultDirectory += "noPrior/";
		else if(loadAspectSentiPrior==1)
			resultDirectory += "aspectSentiPrior/";
		else
			resultDirectory += "aspectPrior/";
		
		String topWordFilePath = resultDirectory+"Topics_"+number_of_topics+"_topWords.txt";
		String infoFilePath = resultDirectory+"Topics_"+number_of_topics+"_Information.txt";
		String featureWeightFilePath = resultDirectory + "Topics_"+number_of_topics+"_FeaturesWeight.csv";;
		
		String debugDirectory = null;
		String debugFilePath = null;
		

		if(debugSentenceTransition){
			debugDirectory = "./debug/"+category+"/"+trainSize+"/";
			if(loadNewEggInTrain)
				debugDirectory += "NewEggLoaded/";
			else
				debugDirectory += "noNewEgg/";
			if(loadAspectSentiPrior==0)
				debugDirectory += "noPrior/";
			else if(loadAspectSentiPrior==1)
				debugDirectory += "aspectSentiPrior/";
			else
				debugDirectory += "aspectPrior/";
			debugFilePath = debugDirectory + "Topics_"+number_of_topics+"_Debug.csv";
			
		}
		
		
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.5; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 30; // Filter the features with DFs smaller than this threshold.
		
		System.out.println("Performing feature selection, wait...");
		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, category,loadProsCons);
		analyzer.LoadStopwords(stopwords);
		
		if(trainSize==0) 
			analyzer.LoadNewEggDirectory(folder, suffix); //Load all the documents as the data set.
		else{
			analyzer.LoadNewEggDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.LoadDirectory(amazonFolder, suffix);
		}
		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		System.out.println("Creating feature vectors, wait...");
		
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			stnModel = "./data/Model/en-sent.bin"; //Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}
		
		analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category,loadProsCons);
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		}
		
		if(trainSize==0) 
			analyzer.LoadNewEggDirectory(folder, suffix); //Load all the documents as the data set.
		else{
			analyzer.LoadNewEggDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.LoadDirectory(amazonFolder, suffix);
		}	
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		
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

			if(generateTrainTestDataForJSTASUM && setRandomFold==false){
				model.setFilePathForJSTASUM(trainSize,category, FilePath);
			}
			
			if(debugSentenceTransition){
				model.setDebugWriter(debugFilePath);
			}
			
			if (crossV<=1) {
				model.EMonCorpus();
				model.printTopWords(topK);
			} else {
				model.setRandomFold(setRandomFold);
				model.crossValidation(crossV);
				model.printTopWords(topK, topWordFilePath);
			}
			
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
			
			if(debugSentenceTransition){
				model.debugOutputWrite();
			}
			if(topicmodel.equals("LRHTSM"))
				((LRHTSM)model).writeOmegaDelta(featureWeightFilePath);
		}
	}
}
