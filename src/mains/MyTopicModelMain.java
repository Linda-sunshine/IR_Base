package mains;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

import structures._Corpus;
import structures._Doc;
import topicmodels.HTMM;
import topicmodels.HTSM;
import topicmodels.LDA_Gibbs;
import topicmodels.LRHTMM;
import topicmodels.LRHTSM;
import topicmodels.ParentChild_Gibbs;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.ParentChildAnalyzer;
import Analyzer.jsonAnalyzer;

public class MyTopicModelMain {
	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each document should have at least 2 sentences
		
		/*****parameters for the two-topic topic model*****/
		String topicmodel = "LRHTMM"; // 2topic, pLSA, HTMM, LRHTMM, Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTSM, ParentChild_Gibbs, ParentChild_GibbsProbitModel
	
		String category = "tablet";
		int number_of_topics = 30;
		boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
		boolean setRandomFold = false; // false means no shuffling and true means shuffling
		int loadAspectSentiPrior = 1; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect 
		
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
		double converge = -1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 20, number_of_iteration = 50, crossV = 1;
		int gibbs_iteration = 2000, gibbs_lag = 50;
//		gibbs_iteration = 10;
//		gibbs_lag = 2;
		double burnIn = 0.4;
		int displayLap = 20;
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
		String articleType = "ArsTech";
		String yahooNewsFolder = "./data/AT-YahooArticles";
		String yahooCommentsFolder = "./data/AT-YahooComments";
		String TechArticlesFolder = "./data/ParentChildTopicModel/ArsTechnicaArticles";
		String TechCommentsFolder = "./data/ParentChildTopicModel/ArsTechnicaComments";
		
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
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_8055.txt", Ngram);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);
		
//		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram, articleType);
//		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_topicmodel_%s.txt", Ngram, articleType);
	
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
		
		File resultFolder = new File(filePrefix);
		if (!resultFolder.exists()) {
			System.out.println("creating directory" + resultFolder);
			resultFolder.mkdir();
		}
		
		////store top k words distribution over topic
		String topWordPath = filePrefix + "/topWords.txt";
		
		/*****Parameters in feature selection.*****/
		String stopwords = "./data/Model/stopwords.dat";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.3; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 5; // Filter the features with DFs smaller than this threshold.

		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel);
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);

		/***** parent child topic model *****/
//		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
		
//		analyzer.LoadParentDirectory(TechArticlesFolder, suffix);
//		analyzer.LoadChildDirectory(TechCommentsFolder, suffix);
		
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM") || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM"))
		{
			analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
			analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		}
		
		String folder = "./data/amazon/small/dedup/RawData";
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.		
		
//		analyzer.LoadNewEggDirectory(newEggFolder, suffix); //Load all the documents as the data set.
//		analyzer.LoadDirectory(amazonFolder, suffix);			
		
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.	
		
		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, converge, beta, c, lambda, analyzer.getBackgroundProb());
			
			if (crossV<=1) {
				for(_Doc d:c.getCollection()) {
					model.inference(d);
					model.printTopWords(topK);
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
			} else if (topicmodel.equals("LDA_Variational")) {		
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
////				model = new LRHTSM_multithread(number_of_iteration, converge, beta, c, 
////						number_of_topics, alpha,
////						lambda);
				model = new LRHTSM(number_of_iteration, converge, beta, c, 
						number_of_topics, alpha,
						lambda);

			} else if (topicmodel.equals("ParentChild_Gibbs")) {
				double mu = 1.0;
				double[] gamma = {2, 2};
				model = new ParentChild_Gibbs(gibbs_iteration, converge, beta-1, c,
						lambda, number_of_topics, alpha-1, burnIn, gibbs_lag,
						gamma, mu);
			}
//				else if(topicmodel.equals("ParentChildWithProbitModel_Gibbs")){
//				double mu = 1.0;
//				double[] gamma = {2, 2};
//				model = new ParentChildWithProbitModel_Gibbs(gibbs_iteration, converge, 
//						beta-1, c, lambda, number_of_topics, alpha-1, burnIn, gibbs_lag, gamma, mu);
//			}
			
			model.setDisplayLap(displayLap);
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
				model.crossValidation(crossV);
				model.printTopWords(topK);
			}
			_Doc d;
			int[][] stnLabels = new int[c.getCollection().size()][];
			if(topicmodel.equals("LRHTMM")){
				for(int i=0; i<c.getCollection().size(); i++){
					d = c.getCollection().get(i);
					stnLabels[i] = ((HTMM) model).get_MAP_topic_assignment(d);
				}
			}
			
			// Write out the sentence labels.
			PrintWriter writer = new PrintWriter(new File("./data/StnLabels.txt"));
			for(int i=0; i<stnLabels.length; i++){
				writer.write(c.getCollection().get(i).getName() + "\t");
				for(int j=0; j<stnLabels[i].length; j++)
					writer.format("%d\t", stnLabels[i][j]);
				writer.write("\n");
			}
			writer.close();
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
