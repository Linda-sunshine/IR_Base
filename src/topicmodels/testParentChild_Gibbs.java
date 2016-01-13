package topicmodels;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.util.Calendar;

import Analyzer.ParentChildAnalyzer;
import structures._Corpus;
import structures._Doc;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;

public class testParentChild_Gibbs {
	public static void main(String[] args) throws IOException, ParseException {
		int classNumber = 5; // Define the number of classes in this Naive
								// Bayes.
		int Ngram = 1; // The default value is unigram.
		String featureValue = "TF"; // The way of calculating the feature value,
									// which can also be "TFIDF", "BM25"
		int norm = 0;// The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; // Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2
											// sentences

		/***** parameters for the two-topic topic model *****/
		// String topicmodel = "CTM_Variational"; // 2topic, pLSA, HTMM, LRHTMM,
		// Tensor, LDA_Gibbs, LDA_Variational, HTSM, LRHTS
		
		String topicmodel = "ParentChild_Gibbs"; //ParentChild_Gibbs, LDA_GibbsParentChild

		String category = "tablet";
		int number_of_topics = 20;
		boolean loadNewEggInTrain = true; // false means in training there is no
											// reviews from NewEgg
		boolean setRandomFold = false; // false means no shuffling and true
										// means shuffling
		int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 =
										// load both senti and aspect; 2 means
										// load only aspect

		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel
				.equals("LDA_Gibbs") ? 200 : 5.0;// these two parameters must be
													// larger than 1!!!
		double converge = 1e-9, lambda = 0.9; // negative converge means do not
												// need to check likelihood
												// convergency
		int varIter = 10;
		double varConverge = 1e-5;
		int topK = 10, number_of_iteration = 50, crossV = 0;
		// int gibbs_iteration = 1500, gibbs_lag = 50;
		int gibbs_iteration = 200, gibbs_lag = 50;
		// int gibbs_iteration = 4, gibbs_lag = 2;
		double burnIn = 0.4;
		boolean display = true, sentence = false;

		// most popular items under each category from Amazon
		// needed for docSummary
		String tabletProductList[] = { "B008GFRDL0" };
		String cameraProductList[] = { "B005IHAIMA" };
		String phoneProductList[] = { "B00COYOAYW" };
		String tvProductList[] = { "B0074FGLUM" };

		/***** The parameters used in loading files. *****/
		// String amazonFolder = "./data/amazon/";
		// String newEggFolder = "./data/NewEgg";
		String articleType = "ArsTech";
		String yahooNewsFolder = "./data/AT-YahooArticles";
		String yahooCommentsFolder = "./data/AT-YahooComments";
		String TechArticlesFolder = "./data/ParentChildTopicModel/ArsTechnicaArticles";
		String TechCommentsFolder = "./data/ParentChildTopicModel/ArsTechnicaComments";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String stnModel = null;
		String posModel = null;
		if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM")
				|| topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM")) {
			stnModel = "./data/Model/en-sent.bin"; // Sentence model.
			posModel = "./data/Model/en-pos-maxent.bin"; // POS model.
			sentence = true;
		}

		String fvFile = String.format(
				"./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram,
				articleType);
		// String fvFile = String
		// .format("./data/Features/fv_1gram_topicmodel.txt");
		// String fvFile = null;
		String fvStatFile = String.format(
				"./data/Features/fv_%dgram_stat_topicmodel.txt", Ngram);

		String aspectList = "./data/Model/aspect_" + category + ".txt";
		String aspectSentiList = "./data/Model/aspect_sentiment_" + category
				+ ".txt";

		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

//		LocalTime today = LocalTime.now();
		Calendar today = Calendar.getInstance();
		
		File resultRootFolder = new File("./data/results");
		if (!resultRootFolder.exists()) {
			System.out.println("creating directory" + resultRootFolder);
			resultRootFolder.mkdir();
		}
		
		String filePrefix = "./data/results/"+today.get(Calendar.DATE)+today.get(Calendar.HOUR_OF_DAY)+topicmodel;
		File resultFolder = new File(filePrefix);
		if (!resultFolder.exists()) {
			System.out.println("creating directory" + resultFolder);
			resultFolder.mkdir();
		}
		
		String infoFilePath = filePrefix+ "/information.txt";

		// /*****Parameters in feature selection.*****/
		// String stopwords = "./data/Model/stopwords.dat";
		// String featureSelection = "DF"; // Feature selection method.
		// double startProb = 0.5; // Used in feature selection, the starting
		// point
		// // of the features.
		// double endProb = 0.999; // Used in feature selection, the ending
		// point
		// // of the features.
		// int DFthreshold = 30; // Filter the features with DFs smaller than
		// this
		// // threshold.
		//

		// System.out.println("Performing feature selection, wait...");
		// jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber,
		// null,
		// Ngram, lengthThreshold);
		// analyzer.LoadStopwords(stopwords);
		// analyzer.LoadDirectory(folder, suffix); // Load all the documents as
		// the
		// // data set.
		// analyzer.featureSelection(fvFile, featureSelection, startProb,
		// endProb,
		// DFthreshold); // Select the features.

		System.out.println("Creating feature vectors, wait...");
		// jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber,
		// fvFile, Ngram, lengthThreshold, stnModel, posModel);
		// newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber,
		// fvFile, Ngram, lengthThreshold, stnModel, posModel, category, 2);
		// if (topicmodel.equals("HTMM") || topicmodel.equals("LRHTMM")
		// || topicmodel.equals("HTSM") || topicmodel.equals("LRHTSM")) {
		// analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
		// analyzer.loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords,
		// pathToNegWords, pathToNegationWords);
		// }

		// analyzer.LoadNewEggDirectory(newEggFolder, suffix); //Load all the
		// documents as the data set.
		// analyzer.LoadDirectory(amazonFolder, suffix);

		/***** Parameters in feature selection for parent child topic model. *****/
		// String stopwords = "./data/Model/stopwords.dat";
		// String featureSelection = "DF"; // Feature selection method.
		//
		// // Used in feature selection, the starting point of the features.
		// double startProb = 0.00;
		//
		// // Used in feature selection, the ending point of the features.
		// double endProb = 0.999;
		//
		// // Filter the features with DFs smaller than this threshold
		// int DFthreshold = 5;
		//
		// System.out.println("Performing feature selection, wait...");
		// ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel,
		// classNumber, null, Ngram, lengthThreshold);
		//
		// analyzer.LoadStopwords(stopwords);
		// // load parent documents
		// analyzer.LoadParentDirectory(TechArticlesFolder, suffix);
		// analyzer.LoadChildDirectory(TechCommentsFolder, suffix);
		// analyzer.featureSelection(fvFile, featureSelection, startProb,
		// endProb,
		// DFthreshold); // Select the features.

		/***** parent child topic model *****/
		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.LoadParentDirectory(TechArticlesFolder, suffix);
		analyzer.LoadChildDirectory(TechCommentsFolder, suffix);
		analyzer.setFeatureValues(featureValue, norm);
		// // Get the collection of all the documents.
		_Corpus c = analyzer.returnCorpus(fvStatFile);

		if (topicmodel.equals("2topic")) {
			twoTopic model = new twoTopic(number_of_iteration, converge, beta,
					c, lambda, analyzer.getBackgroundProb());

			if (crossV <= 1) {
				for (_Doc d : c.getCollection()) {
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
				model = new pLSA_multithread(number_of_iteration, converge,
						beta, c, lambda, number_of_topics, alpha);
			} else if (topicmodel.equals("LDA_Gibbs")) {
				// in gibbs sampling, no need to compute log-likelihood during
				// sampling
				model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, lambda,
						number_of_topics, alpha, burnIn, gibbs_lag);
			} else if (topicmodel.equals("LDA_Variational")) {
				model = new LDA_Variational_multithread(number_of_iteration,
						converge, beta, c, lambda, number_of_topics, alpha,
						varIter, varConverge);
			} else if (topicmodel.equals("HTMM")) {
				model = new HTMM(number_of_iteration, converge, beta, c,
						number_of_topics, alpha);
			} else if (topicmodel.equals("HTSM")) {
				model = new HTSM(number_of_iteration, converge, beta, c,
						number_of_topics, alpha);
			} else if (topicmodel.equals("LRHTMM")) {
				model = new LRHTMM(number_of_iteration, converge, beta, c,
						number_of_topics, alpha, lambda);
			} else if (topicmodel.equals("LRHTSM")) {
				// model = new LRHTSM_multithread(number_of_iteration, converge,
				// beta, c,
				// number_of_topics, alpha,
				// lambda);
				model = new LRHTSM(number_of_iteration, converge, beta, c,
						number_of_topics, alpha, lambda);
			} else if (topicmodel.equals("CTM_Variational")) {
				// model = new CTM_Variational(number_of_iteration, converge,
				// beta, c, lambda, number_of_topics, alpha, varIter,
				// varConverge);
			} else if (topicmodel.equals("ParentChild_Gibbs")) {
				alpha = alpha - 1;
				double mu = 0.001;
				double[] gamma = new double[2];
				gamma[0] = 2;
				gamma[1] = 2;
				model = new ParentChild_Gibbs(gibbs_iteration, 0, beta, c,
						lambda, number_of_topics, alpha, burnIn, gibbs_lag,
						gamma, mu);
			}
			
			model.setDisplay(display);
			model.setInforWriter(infoFilePath);
			// model.setNewEggLoadInTrain(loadNewEggInTrain);

			if (loadAspectSentiPrior == 1) {
				System.out.println("Loading aspect-senti list from "
						+ aspectSentiList);
				model.setSentiAspectPrior(true);
				model.LoadPrior(aspectSentiList, eta);
			} else if (loadAspectSentiPrior == 2) {
				System.out.println("Loading aspect list from " + aspectList);
				model.setSentiAspectPrior(false);
				model.LoadPrior(aspectList, eta);
			} else {
				System.out.println("No prior is added!!");
			}

			String topWordPath = filePrefix+"/topWords.txt";
			// String topWordPath = null;

			if (crossV <= 1) {
				model.EMonCorpus();
				if (topWordPath == null) {
					model.printTopWords(topK);
				} else {
					System.out.println("topK path");
					model.printTopWords(topK, topWordPath);
				}
			} else {
				model.setRandomFold(setRandomFold);
				model.crossValidation(crossV);
				model.printTopWords(topK);
			}
		}

	}
}
