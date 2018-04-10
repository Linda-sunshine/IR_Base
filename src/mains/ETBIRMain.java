package mains;

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

//        /**
//         * generate vocabulary: too large.. ask Lin about it
//         */
//        double startProb = 0.1; // Used in feature selection, the starting point of the features.
//        double endProb = 0.999; // Used in feature selection, the ending point of the features.
//        int maxDF = -1, minDF = 10; // Filter the features with DFs smaller than this threshold.
//        String featureSelection = "IG";
//
//        String folder = "./myData/byUser/";
//        String suffix = ".json";
//        String stopwords = "./data/Model/stopwords.dat";
//        String pattern = String.format("%dgram_%s", Ngram, featureSelection);
//        String fvFile = String.format("data/Features/fv_%s_byUser_20.txt", pattern);
//        String fvStatFile = String.format("data/Features/fv_stat_%s_byUser_20.txt", pattern);
//        String vctFile = String.format("data/Fvs/vct_%s_byUser_20.dat", pattern);
//
////        /****Loading json files*****/
//        DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold);
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
        String fvFile = "./data/Features/fv_2gram_IG_byUser_20.txt";
        String reviewFolder = "./myData/byUser_1/";
        String suffix = ".json";

        ReviewAnalyzer analyzer = new ReviewAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
        analyzer.LoadDirectory(reviewFolder, suffix);

        _Corpus corpus = analyzer.getCorpus();
//        corpus.save2File("./myData/byUser/top20_byUser20.dat");

        int topic_number = 10;

        int varMaxIter = 20;
        double varConverge = 1e-5;

        int emMaxIter = 20;
        double emConverge = 1e-3;


        double alpha =1 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0, lambda=1e-3;//these two parameters must be larger than 1!!!
        double  sigma = 1.0 + 1e-2, rho = 1.0 + 1e-2;

        // LDA
//        /*****parameters for the two-topic topic model*****/
//        String topicmodel = "LDA_Variational"; // pLSA, LDA_Gibbs, LDA_Variational
//
//        int number_of_topics = 15;
//        double converge = -1; // negative converge means do need to check likelihood convergency
//        int number_of_iteration = 100;
//        boolean aspectSentiPrior = true;
//        pLSA tModel = null;
//        if (topicmodel.equals("pLSA")) {
//            tModel = new pLSA_multithread(number_of_iteration, converge, beta, corpus,
//                    lambda, number_of_topics, alpha);
//        } else if (topicmodel.equals("LDA_Gibbs")) {
//            tModel = new LDA_Gibbs(number_of_iteration, converge, beta, corpus,
//                    lambda, number_of_topics, alpha, 0.4, 50);
//        }  else if (topicmodel.equals("LDA_Variational")) {
//            tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, corpus,
//                    lambda, number_of_topics, alpha, 10, -1);
//        } else {
//            System.out.println("The selected topic model has not developed yet!");
//            return;
//        }
//
//        tModel.setDisplayLap(0);
//        tModel.setInforWriter(reviewFolder + topicmodel + "_info.txt");
//        tModel.EMonCorpus();
//        tModel.printTopWords(50, reviewFolder + topicmodel + "_topWords.txt");

        // my model

        ETBIR etbirModel = new ETBIR(emMaxIter, emConverge, beta, corpus, lambda,
                topic_number, alpha, varMaxIter, varConverge, sigma, rho);
        etbirModel.analyzeCorpus();
        etbirModel.EM();
        etbirModel.printTopWords(topic_number, reviewFolder + "topwords-10.txt");
        etbirModel.printEta(reviewFolder + "eta-10.txt");
        etbirModel.printP(reviewFolder + "P-10.txt");
    }
}
