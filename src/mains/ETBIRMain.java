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

        String trainset = "byItem";
        String source = "yelp";
        String dataset = "./myData/" + source + "/" + trainset + "/";

        /**
         * generate vocabulary: too large.. ask Lin about it
         */
//        double startProb = 0.4; // Used in feature selection, the starting point of the features.
//        double endProb = 0.999; // Used in feature selection, the ending point of the features.
//        int maxDF = -1, minDF = 40; // Filter the features with DFs smaller than this threshold.
//        String featureSelection = "IG";
//

//        String suffix = ".json";
//        String stopwords = "./data/Model/stopwords.dat";
//        String pattern = String.format("%dgram_%s", Ngram, featureSelection);
//        String fvFile = String.format("data/Features/fv_%s_" + source + trainset + ".txt", pattern);
//        String fvStatFile = String.format("data/Features/fv_stat_%s_" + source + trainset + ".txt", pattern);
//        String vctFile = String.format("data/Fvs/vct_%s_" + source + trainset + ".dat", pattern);
//
////        /****Loading json files*****/
//        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, source);
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
        fvFiles[0] = "./data/Features/fv_2gram_IG1_byUser_30_50_25.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG_byUser_40_50_12.txt";
        fvFiles[2] = "./data/Features/yelp_features.txt";
        int fvFile_point = 0;

        String reviewFolder = dataset + "data/";
        String outputFolder = dataset + "output/feature_" + fvFile_point + "/";
        String suffix = ".json";
        String topicmodel = "ETBIR"; // pLSA, LDA_Gibbs, LDA_Variational, ETBIR

        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point], Ngram, lengthThreshold, source);
        analyzer.LoadDirectory(reviewFolder, suffix);

        _Corpus corpus = analyzer.getCorpus();
//        corpus.save2File("./myData/byUser/top20_byUser20.dat");

        int number_of_topics = 20;

        int varMaxIter = 10;

        double varConverge = 1e-3;

        int emMaxIter = 100;
        double emConverge = -1;
        double emConverge4ETBIR = 1e-5;


        double alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3;//these two parameters must be larger than 1!!!
        double  sigma = 1.0 + 1e-3, rho = 0.1;

        // LDA
        /*****parameters for the two-topic topic model*****/

        pLSA tModel = null;
        if (topicmodel.equals("pLSA")) {
            tModel = new pLSA_multithread(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha);
        } else if (topicmodel.equals("LDA_Gibbs")) {
            tModel = new LDA_Gibbs(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha, 0.4, 50);
        }  else if (topicmodel.equals("LDA_Variational")) {
            tModel = new LDA_Variational_multithread(emMaxIter, emConverge, beta, corpus,
                    lambda, number_of_topics, alpha, varMaxIter, varConverge); //set this negative!! or likelihood will not change
        } else if (topicmodel.equals("ETBIR")){
            tModel = new ETBIR(emMaxIter, emConverge4ETBIR, beta, corpus, lambda,
                    number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
        }else {
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

//        etbirModel.printEta(outputFolder + "eta.txt");
//        etbirModel.printP(outputFolder + "P.txt");
    }
}
