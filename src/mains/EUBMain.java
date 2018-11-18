package mains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */

public class EUBMain {

    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "YelpNew"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users_1000", prefix, dataset);

        int kFold = 5, k = 0;
        String friendFile = String.format("%s/%s/%sFriends_1000.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex_1000.txt", prefix, dataset, dataset);
        String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, true);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();

        // "cv4edge", "cs4doc"
        String mode = "cv4doc";
        if (mode.equals("cv4doc")) {
            //if it is cv for doc, use all the interactions + part of docs
            analyzer.loadCVIndex(cvIndexFile, kFold);
            analyzer.loadInteractions(friendFile);
        } else if (mode.equals("cv4edge")) {
            // if it is cv for edge, use all the docs + part of edges
            analyzer.loadInteractions(cvIndexFile4Interaction);
        }
        _Corpus corpus = analyzer.getCorpus();

        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 50, number_of_topics = 20, varMaxIter = 10, embeddingDim = 10, innerIter = 1;
        //these two parameters must be larger than 1!!!
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6, stepSize = 1e-3;
        boolean alphaFlag = true, gammaFlag = true, betaFlag = true, tauFlag = true, xiFlag = true, rhoFlag = true;
        boolean multiFlag = true, adaFlag = false;

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;

        if (multiFlag)
            tModel = new EUB_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        else
            tModel = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        ((EUB) tModel).initLookupTables(analyzer.getUsers());
        ((EUB) tModel).setModelParamsUpdateFlags(alphaFlag, gammaFlag, betaFlag, tauFlag, xiFlag, rhoFlag);
        ((EUB) tModel).setMode(mode);

        ((EUB) tModel).setInnerMaxIter(innerIter);
        ((EUB) tModel).setStepSize(stepSize);

        long current = System.currentTimeMillis();
        String saveDir = String.format("./data/embeddingExp/eub/%s_emIter_%d_nuTopics_%d_varIter_%d_innerIter_%d_dim_%d_ada_%b/" +
                "fold_%d_%d", dataset, emMaxIter, number_of_topics, varMaxIter, innerIter, embeddingDim, adaFlag, k, current);
        ((EUB) tModel).fixedCrossValidation(k, saveDir);
        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start) / ((1000 * 60 * 60) * 1.0);
        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));
    }
}

