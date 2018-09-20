package myMains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures.*;
import topicmodels.UserEmbedding.EUB;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

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

        String dataset = "YelpNew"; // "Amazon", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
//		String prefix = "/zf8/lg5bt/DataSigir";

        String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset);
        String userFolder = String.format("%s/%s/Users_1000", prefix, dataset);
        String friendFile = String.format("%s/%s/%sFriends_1000.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex_1000.txt", prefix, dataset, dataset);

        double rho = 0.1;
        int kFold = 5;

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, false, rho);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        String interactionFile = String.format("%s/%s/%sInteractions_1000.txt", prefix, dataset, dataset);
        String nonInteractionFile = String.format("%s/%s/%sNonInteractions_1000.txt", prefix, dataset, dataset);

//        analyzer.loadUserDir(userFolder);
//        analyzer.constructNetwork(friendFile);
//        analyzer.saveNetwork(interactionFile, analyzer.getInteractionMap());
//        analyzer.saveNetwork(nonInteractionFile, analyzer.getNonInteractionMap());
//        analyzer.constructUserIDIndex();
//        analyzer.saveCVIndex(kFold, cvIndexFile);

        // we store the interaction information before-hand, load them directly
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.loadInteractions(interactionFile);
        analyzer.loadNonInteractions(nonInteractionFile);
        analyzer.loadCVIndex(cvIndexFile);

        int emMaxIter = 20, number_of_topics = 20, varMaxIter = 20, embeddingDim = 20;
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();
        EUB eub = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        eub.buildLookupTables(analyzer.getUsers());
//        eub.EMonCorpus();
        eub.fixedCrossValidation(kFold);

        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        System.out.println("[Info]End time: " + end);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60);
        System.out.print(String.format("[Time]This training+testing process took %.2f mins.\n", hours));

    }
}
