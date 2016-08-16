package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import opennlp.tools.util.InvalidFormatException;
import Analyzer.BinaryRouteAnalyzer;
import Classifier.supervised.modelAdaptation.RegLR.MTRegLR;

public class MyRouteMain {
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		int displayLv = 1;

		double trainRatio = 0, adaptRatio = 0.5;
		boolean enforceAdapt = true;

		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		String userFolder = "./data/RouteData/";
		
		BinaryRouteAnalyzer analyzer = new BinaryRouteAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder);
//		analyzer.setFeatureValues("TF", 2);
		analyzer.Normalize(2);
		
		int featureSize = 14;
		MTRegLR adaptation = new MTRegLR(classNumber, featureSize, null, null);
		adaptation.setGlobalModel(14);
		adaptation.setLNormFlag(false);
		adaptation.loadUsers(analyzer.getUsers());
		adaptation.setDisplayLv(displayLv);
		adaptation.train();
		adaptation.test();
	}
}
