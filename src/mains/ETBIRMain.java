package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import Analyzer.MultiThreadedReviewAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.ReviewAnalyzer;
import structures._Corpus;
import structures._Doc;
import structures._Review;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.LDA.LDA_Variational;
import Analyzer.BipartiteAnalyzer;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.embeddingModel.ETBIR_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import utils.Utils;


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

        /*****parameters for topic model*****/
        int number_of_topics = 20;
        int varMaxIter = 30;
        double varConverge = 1e-4;
        int emMaxIter = 50;
        double emConverge = 1e-12;

        double alpha = 1+1e-1, beta = 1 + 1e-3, lambda = 1 + 1e-3;//these two parameters must be larger than 1!!!
        double sigma = 1.01, rho = 1.01;

        int topK = 50;
        int crossV = 2;
        boolean setRandomFold = true;

        /*****data setting*****/
        String trainset = "byUser_4k_review";
        String source = "yelp";
        String dataset = "./myData/" + source + "/" + trainset + "/";

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

        String reviewFolder = dataset + "data/"; //2foldsCV/folder0/train/, data/
        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point],
                Ngram, lengthThreshold, numberOfCores, true, source);
        _Corpus corpus = new _Corpus();
        if(crossV>1 && setRandomFold==false){
            reviewFolder = dataset + crossV + "foldsCV/";
        }else{
//        analyzer.setReleaseContent(false);//Remember to set it as false when generating crossfolders!!!
            analyzer.loadUserDir(reviewFolder);
            corpus = analyzer.getCorpus();
        }

        String outputFolder = dataset + "output/" + crossV + "foldsCV" + "/";
        String suffix = ".json";

//        corpus.save2File(dataset + "yelp_40_50_12.dat");//for CTM

//        BipartiteAnalyzer cv = new BipartiteAnalyzer(corpus); // split corpus into folds
//        cv.analyzeCorpus();
//        cv.splitCorpus(crossV,dataset + crossV + "foldsCV/");


        /*****model loading*****/
        String topicmodel = "ETBIR"; // pLSA, LDA_Gibbs, LDA_Variational, ETBIR
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
            tModel = new ETBIR_multithread(emMaxIter, emConverge, beta, corpus, lambda,
                    number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
        }else {
            System.out.println("The selected topic model has not developed yet!");
            return;
        }

        tModel.setDisplayLap(1);
        new File(outputFolder).mkdirs();
        tModel.setInforWriter(outputFolder + topicmodel + "_info.txt");
        if (crossV<=1) {//just train
            tModel.EMonCorpus();
            tModel.printTopWords(topK, outputFolder + topicmodel + "_topWords.txt");
            tModel.printParameterAggregation(topK, outputFolder, topicmodel);
            tModel.closeWriter();
        } else if(setRandomFold == true){//cross validation with random folds
            tModel.setRandomFold(setRandomFold);
            double trainProportion = ((double)crossV - 1)/(double)crossV;
            double testProportion = 1-trainProportion;
            tModel.setPerplexityProportion(testProportion);
            tModel.crossValidation(crossV);
            tModel.printTopWords(topK, outputFolder + topicmodel + "_topWords.txt");
        } else{//cross validation with fixed folds
            double[] perf = new double[crossV];
            for(int k = 0; k <crossV; k++){
                analyzer.reset();
                //load test set
                String testFolder = reviewFolder + k + "/";
                analyzer.loadUserDir(testFolder);
                for(_Doc d : analyzer.getCorpus().getCollection()){
                    ((_Review)d).setType(_Review.rType.TEST);
                }
                //load train set
                for(int i = 0; i < crossV; i++){
                    if(i!=k){
                        String trainFolder = reviewFolder + i + "/";
                        analyzer.loadUserDir(trainFolder);
                    }
                }

                tModel.setCorpus(analyzer.getCorpus());
                perf[k] = tModel.oneFoldValidation();

                String resultFolder = outputFolder + k + "/";
                new File(resultFolder).mkdirs();
                tModel.printParameterAggregation(topK, resultFolder, topicmodel);
            }

            //output the performance statistics
            double mean = Utils.sumOfArray(perf)/crossV, var = 0;
            for(int i=0; i<perf.length; i++)
                var += (perf[i]-mean) * (perf[i]-mean);
            var = Math.sqrt(var/crossV);
            System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);
        }

    }
}
