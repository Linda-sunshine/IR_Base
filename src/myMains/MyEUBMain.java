package myMains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.MultiThreadedTADWAnalyzer;
import Analyzer.UserAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import structures._User;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.UserEmbedding.EUB4ColdStart_multithreading;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The main entrance for calling EUB model
 */
public class MyEUBMain {

    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "StackOverflow"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        int kFold = 5, k = -1;
        int time = 2;
//        for(int time: new int[]{2, 3, 4, 5, 6, 7, 8}) {
//        /***Feature selection**/
//        UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, true);
//        analyzer.LoadStopwords("./data/Model/stopwords.dat");
//        analyzer.loadUserDir(userFolder);
//        analyzer.featureSelection("./data/StackOverflow_DF_10k.txt", "DF", 10000, 100, 5000);

        String orgFriendFile = String.format("%s/%s/%sFriends_org.txt", prefix, dataset, dataset);
        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
//        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);
        String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction_time_%d.txt", prefix, dataset, dataset, time);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        /***print data for TADW***/
//        String tadwUserFile = String.format("./data/CoLinAdapt/%s/%sUsers4TADW.txt", dataset, dataset);
//        analyzer.setReleaseContent(false);
//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.loadCVIndex(cvIndexFile, kFold);
//        analyzer.writeAggregatedUsers(tadwUserFile, kFold);
//
//        MultiThreadedTADWAnalyzer tadwanalyzer = new MultiThreadedTADWAnalyzer(tokenModel, classNumber, providedCV,
//                Ngram, lengthThreshold, numberOfCores, true);
//        tadwanalyzer.loadUserTxt(tadwUserFile, kFold, k);
//        tadwanalyzer.setFeatureValues("TFIDF", 1);
//        tadwanalyzer.printData4TADW(String.format("./data/CoLinAdapt/%s/%sTADW_text_fold_%d.txt", dataset, dataset, k));

        /****save cv index for documents before-hand****/
//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.saveCVIndex(5, cvIndexFile);
//        analyzer.loadInteractions(orgFriendFile);
//        analyzer.saveNetwork(friendFile);

        /****save cv index for interactions before-hand****/
//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.loadCVIndex(cvIndexFile, kFold);
//        analyzer.loadInteractions(friendFile);
//        analyzer.assignCVIndex4Network(kFold, time);
//        analyzer.sanityCheck4CVIndex4Network(true);
//        analyzer.sanityCheck4CVIndex4Network(false);
//        if(time == 2)
//            analyzer.saveCVIndex4Network(cvIndexFile4Interaction, true);
//        analyzer.saveCVIndex4Network(cvIndexFile4NonInteraction, false);


        /***Our algorithm EUB****/
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        String mode = "cv4doc"; // "cv4edge" "cs4doc"--"cold start for doc" "cs4edge"--"cold start for edge"
        boolean coldStartFlag = false;

        //if it is cv for doc, use all the interactions + part of docs
        if(mode.equals("cv4doc") && !coldStartFlag){
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        // if it is cv for edge, use all the docs + part of edges
        } else if(mode.equals("cv4edge") && !coldStartFlag){
            analyzer.loadInteractions(cvIndexFile4Interaction);
        // cold start for doc, use all edges, test doc perplexity on light/medium/heavy users
        } else if(mode.equals("cv4doc") && coldStartFlag){
            cvIndexFile = String.format("%s/%s/ColdStart/%s_cold_start_4docs_fold_%d.txt", prefix, dataset, dataset, k);
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        // cold start for edge, use all edges, learn user embedding for light/medium/heavy users
        } else if(mode.equals("cv4edge") && coldStartFlag){
            cvIndexFile4Interaction = String.format("%s/%s/ColdStart/%s_cold_start_4edges_fold_%d_interactions.txt", prefix,
                    dataset, dataset, k);
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        _Corpus corpus = analyzer.getCorpus();

        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 100, number_of_topics = 30, varMaxIter = 10, embeddingDim = 10, innerIter = 1, inferIter = 3;
        //these two parameters must be larger than 1!!!
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6, stepSize = 1e-3;
        boolean alphaFlag = true, gammaFlag = true, betaFlag = true, tauFlag = true, xiFlag = true;
        boolean multiFlag = true, adaFlag = false;

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;

        if(multiFlag && coldStartFlag)
            tModel = new EUB4ColdStart_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        else if(multiFlag && !coldStartFlag)
            tModel = new EUB_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        else
            tModel = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        ((EUB) tModel).initLookupTables(analyzer.getUsers());
        ((EUB) tModel).setModelParamsUpdateFlags(alphaFlag, gammaFlag, betaFlag, tauFlag, xiFlag);
        ((EUB) tModel).setMode(mode);

        ((EUB) tModel).setInnerMaxIter(innerIter);
        ((EUB) tModel).setInferMaxIter(inferIter);
        ((EUB) tModel).setStepSize(stepSize);

        long current = System.currentTimeMillis();
        String saveDir = String.format("./data/emebddingExp/eub/%s_emIter_%d_nuTopics_%d_varIter_%d_innerIter_%d_dim_%d_ada_%b/" +
                "fold_%d_%d", dataset, emMaxIter, number_of_topics, varMaxIter, innerIter, embeddingDim, adaFlag, kFold, current);

        if(multiFlag && coldStartFlag)
            ((EUB4ColdStart_multithreading) tModel).fixedCrossValidation(k, saveDir);
        else
            ((EUB) tModel).fixedCrossValidation(k, saveDir);
        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60*60);
        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));

    }
}