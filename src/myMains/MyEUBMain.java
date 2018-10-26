package myMains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.UserAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.FileNotFoundException;
import java.io.IOException;

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

        String dataset = "YelpNew"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/yelp_features.txt", prefix, dataset);
        String userFolder = String.format("%s/%s/Users_1000", prefix, dataset);

        int kFold = 5, k = 0, time = 2;
//        /***Feature selection**/
//        UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, true);
//        analyzer.LoadStopwords("./data/Model/stopwords.dat");
//        analyzer.loadUserDir(userFolder);
//        analyzer.featureSelection("./data/YelpNew_DF_3k.txt", "DF", 7000, 100, 3000);

        String orgFriendFile = String.format("%s/%s/%sFriends_org.txt", prefix, dataset, dataset);
        String friendFile = String.format("%s/%s/%sFriends_1000.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex_1000.txt", prefix, dataset, dataset);
        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
        String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction_time_%d.txt", prefix, dataset, dataset, time);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.printData4TADW(String.format("./data/%sTADW.txt", dataset));

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
////            analyzer.saveCVIndex4Network(cvIndexFile4Interaction, true);
//        analyzer.saveCVIndex4Network(cvIndexFile4NonInteraction, false);

        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();

        String mode = "cv4doc"; // "cv4edge"
        // if it is cv for doc, use all the interactions + part of docs
        if(mode.equals("cv4doc")){
            analyzer.loadInteractions(friendFile);
            analyzer.loadCVIndex(cvIndexFile, kFold);
        // if it is cv for edge, use all the docs + part of edges
        } else if(mode.equals("cv4edge")){
            // if no cv index file for doc is loaded, then all documents will be training docs.
            cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        _Corpus corpus = analyzer.getCorpus();

        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 50, number_of_topics = 20, varMaxIter = 10, embeddingDim = 10, innerIter = 1, inferIter = 3;
        //these two parameters must be larger than 1!!!
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6, stepSize = 1e-3;
        boolean alphaFlag = true, gammaFlag = true, betaFlag = true, tauFlag = true, xiFlag = true, multiFlag = true, adaFlag = false;

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;
        if(multiFlag)
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
        ((EUB) tModel).fixedCrossValidation(k, saveDir);
        long end = System.currentTimeMillis();

        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60*60);
        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));

    }
}
