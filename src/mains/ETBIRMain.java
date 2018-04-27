package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import Analyzer.ReviewAnalyzer;
import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;


/**
 * @author Lu Lin
 */
public class ETBIRMain {

    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        String tokenModel = "./data/Model/en-token.bin";

        /**
         * generate vocabulary: too large.. ask Lin about it
         */
//        double startProb = 0.2; // Used in feature selection, the starting point of the features.
//        double endProb = 0.999; // Used in feature selection, the ending point of the features.
//        int maxDF = -1, minDF = 30; // Filter the features with DFs smaller than this threshold.
//        String featureSelection = "IG";
//
//        String trainset = "byUser_30_50_25";
//        String folder = "./myData/" + trainset + "/";
//        String suffix = ".json";
//        String stopwords = "./data/Model/stopwords.dat";
//        String pattern = String.format("%dgram_%s", Ngram, featureSelection);
//        String fvFile = String.format("data/Features/fv_%s_" + trainset + ".txt", pattern);
//        String fvStatFile = String.format("data/Features/fv_stat_%s_" + trainset + ".txt", pattern);
//        String vctFile = String.format("data/Fvs/vct_%s_" + trainset + ".dat", pattern);
//
////        /****Loading json files*****/
//        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
//        analyzer.LoadStopwords(stopwords);
//        analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//
////		/****Feature selection*****/
//        System.out.println("Performing feature selection, wait...");
//        analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, maxDF, minDF); //Select the features.
//        analyzer.SaveCVStat(fvStatFile);

        /**
         * model training
         */
        String[] fvFiles = new String[3];
//        fvFiles[0] = "./data/Features/fv_2gram_IG1_byUser_30_50_25.txt";
        fvFiles[0] = "./data/Features/fv_2gram_IG_byUser_20.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG2_byUser_30_50_25.txt";
        fvFiles[2] = "./data/Features/yelp_features.txt";
        int fvFile_point = 0;
        String dataset = "./myData/byUser_40_50_12";
        String reviewFolder = dataset + "/data/";
        String outputFolder = dataset + "/output/feature2_" + fvFile_point + "/";
        String suffix = ".json";

        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point], Ngram, lengthThreshold);
        analyzer.LoadDirectory(reviewFolder, suffix);

        _Corpus corpus = analyzer.getCorpus();
//        corpus.save2File("./myData/byUser/top20_byUser20.dat");

        int number_of_topics = 20;

        int varMaxIter = 10;
        double varConverge = 1e-4;

        int emMaxIter = 100;
        double emConverge = 1e-6;


        double alpha = 1 + 1e-2, beta = 1 + 1e-3, eta = 5.0, lambda = 1 + 1e-3;//these two parameters must be larger than 1!!!
        double  sigma = 1.0 + 1e-2, rho = 1.0 + 1e-2;

        // LDA
        /*****parameters for the two-topic topic model*****/
        String topicmodel = "LDA_Variational"; // pLSA, LDA_Gibbs, LDA_Variational

        pLSA tModel = null;
        if (topicmodel.equals("pLSA")) {
            tModel = new pLSA_multithread(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha);
        } else if (topicmodel.equals("LDA_Gibbs")) {
            tModel = new LDA_Gibbs(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha, 0.4, 50);
        }  else if (topicmodel.equals("LDA_Variational")) {
            tModel = new LDA_Variational_multithread(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha, 10, -1); //set this negative!! or likelihood will not change
        } else {
            System.out.println("The selected topic model has not developed yet!");
            return;
        }

        tModel.setDisplayLap(1);
        new File(outputFolder).mkdirs();
        tModel.setInforWriter(outputFolder + topicmodel + "_info.txt");
        tModel.EMonCorpus();
        tModel.printTopWords(50, outputFolder + topicmodel + "_topWords.txt");
        tModel.printParameterAggregation(50, outputFolder, topicmodel);
        tModel.closeWriter();

        // my model
//        ETBIR etbirModel = new ETBIR(emMaxIter, emConverge, beta, corpus, lambda,
//                number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
//        etbirModel.analyzeCorpus();
//        etbirModel.EM();
//        etbirModel.printTopWords(number_of_topics, outputFolder + "topwords.txt");
//        etbirModel.printEta(outputFolder + "eta.txt");
//        etbirModel.printP(outputFolder + "P.txt");
    }
}
