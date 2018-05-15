package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import Analyzer.MultiThreadedReviewAnalyzer;
import Analyzer.ReviewAnalyzer;
import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.embeddingModel.ETBIR_multithread;
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
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        String tokenModel = "./data/Model/en-token.bin";

        String trainset = "byUser_40_50_12";
        String source = "yelp";
        String dataset = "./myData/" + source + "/" + trainset + "/";

        /**
         * model training
         */
        String[] fvFiles = new String[4];
        fvFiles[0] = "./data/Features/fv_2gram_IG_yelp_byUser_30_50_25.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG_amazon_movie_byUser_40_50_12.txt";
        fvFiles[2] = "./data/Features/fv_2gram_IG_amazon_electronic_byUser_20_20_5.txt";
        fvFiles[3] = "./data/Features/fv_2gram_IG_amazon_book_byUser_40_50_12.txt";
        int fvFile_point = 0;
        if(source.equals("amazon_movie")){
            fvFile_point = 1;
        }else if(source.equals("amazon_electronic")){
            fvFile_point = 2;
        }else if(source.equals("amazon_book")){
            fvFile_point = 3;
        }

        String reviewFolder = dataset + "data/";
        String outputFolder = dataset + "output/feature_multi_" + fvFile_point + "/";
        String suffix = ".json";
        String topicmodel = "ETBIR"; // pLSA, LDA_Gibbs, LDA_Variational, ETBIR

        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point],
                Ngram, lengthThreshold, numberOfCores, true, source);
        analyzer.loadUserDir(reviewFolder);
        _Corpus corpus = analyzer.getCorpus();

        //        corpus.save2File("./myData/byUser/top20_byUser20.dat");

        int number_of_topics = 20;

        int varMaxIter = 30;
        double varConverge = 1e-4;

        int emMaxIter = 200;
        double emConverge = -1;
        double emConverge4ETBIR = 1e-5;


        double alpha = 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3;//these two parameters must be larger than 1!!!
        double  sigma = 1e-2, rho = 1e-2;

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
            tModel = new ETBIR_multithread(emMaxIter, emConverge4ETBIR, beta, corpus, lambda,
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

    }
}
