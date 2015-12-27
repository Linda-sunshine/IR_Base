package mains;

import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import Analyzer.jsonAnalyzer;
import clustering.KMeansAlg;

public class ClusteringMain {

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM
		int clusterSize = 50;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/tablet/topicmodel";
		String suffix = ".json";
		String stopword = "./data/Model/stopwords.dat";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
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
		boolean releaseContent = false;
		
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
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		//Added by Mustafizur----------------
		analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
		analyzer.LoadStopwords(stopword); // Load the sentiwordnet file.
		analyzer.loadPriorPosNegWords(sentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		analyzer.setReleaseContent(releaseContent);
		
		// Added by Mustafizur----------------
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
		analyzer.setFeatureValues("BM25", 2);		
		_Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		
		KMeansAlg alg = new KMeansAlg(c, clusterSize);
		alg.train(c.getCollection());
	}

}
