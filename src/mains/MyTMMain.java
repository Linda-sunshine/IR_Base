package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.DocAnalyzer;

public class MyTMMain {
	
	//In the main function, we want to input the data and do adaptation 
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 2;
		String stopwords = "./data/Model/stopword_tm.txt";
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		//int maxDF = -1, minDF = 20; // Filter the features with DFs smaller than this threshold.
		String fs = "CHI";//"IG_CHI"
		int maxDF = -1, minDF = 10; // Filter the features with DFs smaller than this threshold.
		String folder = "/home/lin/Downloads/yelp/all/";

		DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, 1, 0); 
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, "json");
		analyzer.featureSelection("./data/fs_tm/", fs, maxDF, minDF, 10000);
	}
}
