package mains;

import java.io.FileNotFoundException;
import java.io.IOException;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.MultiThreadedUserAnalyzer;

public class MyCategoryCheckMain {
	public static void main(String [] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int displayLv = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		double eta1 = 0.05, eta2 = 0.05, eta3 = 0.05, eta4 = 0.05;
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		String featureGroupFile = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String featureGroupFileB = String.format("./data/CoLinAdapt/%s/CrossGroups_800.txt", dataset);
		String globalModel = String.format("./data/CoLinAdapt/%s/GlobalWeights.txt", dataset);
		String dir = "./data/";

		CategoryAnalyzer analyzer = new CategoryAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadCategory("./data/category.txt");
		analyzer.loadUserDir(userFolder);
	}
}
