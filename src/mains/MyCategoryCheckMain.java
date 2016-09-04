package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;
import structures._Doc;
import structures._Review;
import Analyzer.CategoryAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.CtgSVM;

public class MyCategoryCheckMain {
	public static void main(String [] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		
		CategoryAnalyzer analyzer = new CategoryAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.setReleaseContent(false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadCategory("./data/category.txt");
		analyzer.loadUserDir(userFolder);
		
		HashMap<Integer, ArrayList<_Review>> ctgRvws = analyzer.getCtgRvws();
		/**5 -Electronics
		 * 19-home & kitchen
		 * 24-Books
		 * 25-Amazon Instant Video
		 * 38-movie & TV 
		**/
		// Samples from electronics.
		ArrayList<_Review> electronics = analyzer.sample(ctgRvws.get(5), 2500);
		analyzer.split(electronics, 500);
		ArrayList<_Doc> eletrainSet = analyzer.getTrainSet();
		ArrayList<_Doc> eletestSet = analyzer.getTestSet();
		
		// Samples from home kitchen.
		ArrayList<_Review> homeKitchen = analyzer.sample(ctgRvws.get(19), 2500);
		analyzer.split(homeKitchen, 500);
		ArrayList<_Doc> hmtrainSet = analyzer.getTrainSet();
		ArrayList<_Doc> hmtestSet = analyzer.getTestSet();
		
//		CtgSVM hmSvm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
//		hmSvm.setTrainTestSets(hmtrainSet, eletestSet);
//		hmSvm.train();
//		hmSvm.test();
		
		// Samples from other three categories.
		ArrayList<_Review> books = analyzer.sample(ctgRvws.get(24), 2500);
		analyzer.split(books, 500);
		ArrayList<_Doc> booktrainSet = analyzer.getTrainSet();
		ArrayList<_Doc> booktestSet = analyzer.getTestSet();
		
		
		ArrayList<_Review> amazonvideo = analyzer.sample(ctgRvws.get(25), 2500);
		analyzer.split(amazonvideo, 500);
		ArrayList<_Doc> videotrainSet = analyzer.getTrainSet();
		ArrayList<_Doc> videotestSet = analyzer.getTestSet();
		
		ArrayList<_Review> movietv = analyzer.sample(ctgRvws.get(38), 2500);
		analyzer.split(electronics, 500);
		ArrayList<_Doc> movietrainSet = analyzer.getTrainSet();
		ArrayList<_Doc> movietestSet = analyzer.getTestSet();
		
		// Samples from the five categories.
		ArrayList<_Review> mixed = analyzer.sample(electronics, 400);
		mixed.addAll(analyzer.sample(homeKitchen, 400));
		mixed.addAll(analyzer.sample(books, 400));
		mixed.addAll(analyzer.sample(movietv, 400));
		ArrayList<_Doc> mixedTrainSet = new ArrayList<_Doc>(mixed);
		
		// eletrainSet, hmtrainSet, booktrainSet, videotrainSet, movietrainSet 
		ArrayList<_Doc> testSet = new ArrayList<_Doc>();
		testSet.addAll(eletestSet);
//		testSet.addAll(hmtestSet);
//		testSet.addAll(booktestSet);
//		testSet.addAll(videotestSet);
//		testSet.addAll(movietestSet);
		
		PrintWriter writer = new PrintWriter(new File("weights.xls"));
		writer.write("bias\t");
		for(String s: analyzer.getFeatures())
			writer.write(s+"\t");
		writer.write("\n");
		
		CtgSVM svm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
		svm.setTrainTestSets(eletrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		
		svm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
		svm.setTrainTestSets(hmtrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		
		svm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
		svm.setTrainTestSets(booktrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		
		svm.setTrainTestSets(videotrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		
		svm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
		svm.setTrainTestSets(movietrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		
		svm = new CtgSVM(classNumber, analyzer.getFeatureSize(), 1);
		svm.setTrainTestSets(mixedTrainSet, testSet);
		svm.train();
		svm.test();
		for(double w: svm.getWeights())
			writer.write(w+"\t");
		writer.write("\n");
		writer.close();
		
	}
}
