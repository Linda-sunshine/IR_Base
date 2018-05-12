package myMains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import Analyzer.MultiThreadedReviewAnalyzer;


public class MyETBIRMain {

    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        int lengthThreshold = 5; //Document length threshold
		int numberOfCores = Runtime.getRuntime().availableProcessors();

        String tokenModel = "./data/Model/en-token.bin";
        String fvFile = "./data/Features/fv_2gram_IG_yelp_byUser_30_50_25.txt";
        String reviewFolder = "./data/myData/byUser_40_50_12/data";
        String source = "yelp";
        
		System.out.println("[Info] Start preprocess textual data...");
        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, numberOfCores, true, source);
        analyzer.loadUserDir(reviewFolder);
        _Corpus corpus = analyzer.getCorpus();

        int varMaxIter = 5;
        double varConverge = 1e-5;
        double emConverge = 1e-9;
        double alpha = 1e-2, beta = 1.0 + 1e-3, eta = 200, lambda=1e-3;//these two parameters must be larger than 1!!!
        double  sigma = 1.0 + 1e-2, rho = 1.0 + 1e-2;

        // LDA
        /*****parameters for the two-topic topic model*****/
        String topicmodel = "ETBIR"; // pLSA, LDA_Gibbs, LDA_Variational
        int number_of_topics = 30;
        double converge = -1; // negative converge means do not need to check likelihood convergency
        int number_of_iteration = 30;

        pLSA tModel = null;
        long current = System.currentTimeMillis();
        		
        String resultDir = String.format("./data/result/ETBIR_%d/", current);

        File resultFolder = new File("./data/ETBIR_"+current);
        if (!resultFolder.exists()) {
            System.out.println("[Info]Create directory " + resultFolder);
            resultFolder.mkdir();
        }

        if (topicmodel.equals("pLSA")) {
            tModel = new pLSA_multithread(number_of_iteration, converge, beta, corpus,
                    lambda, number_of_topics, alpha);
        } else if (topicmodel.equals("LDA_Gibbs")) {
            tModel = new LDA_Gibbs(number_of_iteration, converge, beta, corpus,
                    lambda, number_of_topics, alpha, 0.4, 50);
        }  else if (topicmodel.equals("LDA_Variational")) {
            tModel = new LDA_Variational_multithread(number_of_iteration, converge, beta, corpus,
                    lambda, number_of_topics, alpha, 10, 1e-5);
        } else if(topicmodel.equals("ETBIR")){
            tModel = new ETBIR(number_of_iteration, emConverge, beta, corpus, lambda,
            		number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
            ((ETBIR) tModel).analyzeCorpus();
            ((ETBIR) tModel).initial();
        } else{
        	 System.out.println("The selected topic model has not developed yet!");
             return;
        }

        int topk = 30;
        tModel.setDisplayLap(10);
        tModel.EMonCorpus();
        tModel.printTopWords(topk);
        tModel.printTopWords(topk, String.format("./data/%s_topic_topwords_%d.txt", topicmodel, current));
        ((ETBIR) tModel).printParameterAggregation(topk, resultFolder+"/", topicmodel);
    }
}
