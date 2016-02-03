package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import SanityCheck.AnnotatedSanityCheck;
import SanityCheck.BaseSanityCheck;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import Analyzer.Analyzer;
import Analyzer.AspectAnalyzer;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;

/****
 * In this main function, I will apply learning to rank models on the human annotated different groups of reviews 
 * to see if learning to rank can benefit from the grouping.  
 * @author lin
 */
public class MyL2RSanityCheck {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 5;
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		int minimunNumberofSentence = 2; // each sentence should have at least 2 sentences for HTSM, LRSHTM
		int number_of_topics = 30;
		
		//Added by Mustafizur----------------
		String pathToPosWords = "./data/Model/SentiWordsPos.txt";
		String pathToNegWords = "./data/Model/SentiWordsNeg.txt";
		String pathToNegationWords = "./data/Model/negation_words.txt";
		String pathToSentiWordNet = "./data/Model/SentiWordNet_3.0.0_20130122.txt";

		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.

		String stnModel = "./data/Model/en-sent.bin"; //Sentence model. Need it for postagging.
		String stopword = "./data/Model/stopwords.dat";
		String tagModel = "./data/Model/en-pos-maxent.bin";
		
		String category = "tablets"; //"electronics"
		String dataSize = "86jsons"; //"50K", "100K"
		String fvFile = String.format("./data/Features/fv_%dgram_%s_%s.txt", Ngram, category, dataSize);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, category, dataSize);
		String stopwords = "./data/Model/stopwords.dat";
		
		int numOfAspects = 28; // 12, 14, 24, 28
		String aspectlist = String.format("./data/Model/%d_aspect_tablet.txt", numOfAspects);
		String topicFile = String.format("./data/TopicVectors/%dAspects_topicVectors_corpus.txt", numOfAspects);
		
		System.out.println("Creating feature vectors, wait...");
		Analyzer analyzer;
		_Corpus c;
		analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold, tagModel, aspectlist, true);		
		((DocAnalyzer) analyzer).setReleaseContent(false);
		analyzer.setMinimumNumberOfSentences(minimunNumberofSentence);
		((DocAnalyzer) analyzer).loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
//		((DocAnalyzer) analyzer).LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.loadTopicVectors(topicFile, number_of_topics);
		
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		c.mapLabels(4);
		double[][] MAPs = new double[3][];
//		AnnotatedSanityCheck check = new AnnotatedSanityCheck(c, "SVM", 1, 100, BaseSanityCheck.SimType.ST_BoW);
//		check.loadAnnotatedFile("./data/Selected100Files/100Files_IDs_Annotation.txt");
//		check.diffGroupLOOCV();
//		MAPs[0] = check.getMAPs();
//		
//		check = new AnnotatedSanityCheck(c, "SVM", 1, 100, BaseSanityCheck.SimType.ST_TP);
//		check.loadAnnotatedFile("./data/Selected100Files/100Files_IDs_Annotation.txt");
//		check.diffGroupLOOCV();
//		MAPs[1] = check.getMAPs();

		AnnotatedSanityCheck check = new AnnotatedSanityCheck(c, "SVM", 1, 100, BaseSanityCheck.SimType.ST_L2R);
		check.loadAnnotatedFile("./data/Selected100Files/100Files_IDs_Annotation.txt");
		check.diffGroupLOOCV();
		MAPs[2] = check.getMAPs();
		
		String[] methods = new String[]{"BoW", "Topic", "L2R"};
		for(int i=0; i<3; i++){
			System.out.print(methods[i]+":\t");
			for(double m: MAPs[i])
				System.out.print(m+"\t");
			System.out.println();
		}
	}
}
