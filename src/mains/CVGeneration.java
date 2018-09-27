package mains;

import Analyzer.MultiThreadedReviewAnalyzer;
import Analyzer.ReviewAnalyzer;

import java.io.IOException;
import java.text.ParseException;

public class CVGeneration {
    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        String tokenModel = "./data/Model/en-token.bin";
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String trainset = "sample_7k";
        String source = "stackoverflow";
        String dataset = "/zf18/ll5fy/lab/dataset/" + source + "/" + trainset + "/";

        /**
         * generate vocabulary:
         */
        double startProb = 0.2; // Used in feature selection, the starting point of the features.
        double endProb = 0.9; // Used in feature selection, the ending point of the features.
        int maxDF = 9000, minDF = 40; // Filter the features with DFs smaller than this threshold.
        String featureSelection = "IG";


        String suffix = ".txt";
        String stopwords = "./data/Model/stopwords.dat";
        String pattern = String.format("%dgram_%s", Ngram, featureSelection);
        String fvFile = String.format("data/Features/fv_%s_" + source + "_" + trainset + ".txt", pattern);
        String fvStatFile = String.format("data/Features/fv_stat_%s_" + source + trainset + ".txt", pattern);
        String vctFile = String.format("data/Fvs/vct_%s_" + source + trainset + ".dat", pattern);

//        /****Loading json files*****/
        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, true, source);
        analyzer.LoadStopwords(stopwords);
        analyzer.LoadDirectory(dataset, suffix); //Load all the documents as the data set.

//		/****Feature selection*****/
        System.out.println("Performing feature selection, wait...");
        analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
        analyzer.SaveCVStat(fvStatFile);

    }
}
