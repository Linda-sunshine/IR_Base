package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import Analyzer.Analyzer;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;

public class PuritySanityCheckMain {
	//In this main function, I want to check the purity given by Random, BoW, Topic vectors, BoW+Topic 
	//to see in which situation the similarity works and why? 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold		
		int number_of_topics = 30;
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.

		String category = "tablets"; //"electronics"
		String dataSize = "86jsons"; //"50K", "100K"
		String fvFile = String.format("./data/Features/fv_%dgram_%s_%s.txt", Ngram, category, dataSize);
		String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, category, dataSize);
				
		System.out.println("Creating feature vectors, wait...");
		Analyzer analyzer;
		_Corpus c;
		analyzer = new jsonAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
		String topicFile = "./data/topicVectors";
		analyzer.loadTopicVectors(topicFile, number_of_topics);
		
		//construct effective feature values for supervised classifiers 
		analyzer.setFeatureValues("BM25", 2);
		c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.
		c.mapLabels(3);
		
		PuritySanityCheck checkRandom = new PuritySanityCheck(c);
		checkRandom.calculateSimilarity();
		checkRandom.calculatePatK4All();
		checkRandom.printPatK();
		
		PuritySanityCheck checkBow = new PuritySanityCheck(1, c);
		checkBow.calculateSimilarity();
		checkBow.calculatePatK4All();
		checkBow.printPatK();
		
		PuritySanityCheck checkTopic = new PuritySanityCheck(2, c);
		checkTopic.calculateSimilarity();
		checkTopic.calculatePatK4All();
		checkTopic.printPatK();
		
		PuritySanityCheck checkBoWTop = new PuritySanityCheck(3, c);
		checkBoWTop.calculateSimilarity();
		checkBoWTop.calculatePatK4All();
		checkBoWTop.printPatK();		
	}
}
