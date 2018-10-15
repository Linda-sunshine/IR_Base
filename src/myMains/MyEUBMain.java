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
        String size = "1000"; // 10k users-> "_10k"
//		String prefix = "/zf8/lg5bt/DataSigir";

        String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset);
        String userFolder = String.format("%s/%s/Users_%s", prefix, dataset, size);
        String friendFile = String.format("%s/%s/%sFriends_%s.txt", prefix, dataset, dataset, size);
        String cvIndexFile = String.format("%s/%s/%sCVIndex_%s.txt", prefix, dataset, dataset, size);
        String userIdFile = String.format("%s/%s/%sUserIds_%s.txt", prefix, dataset, dataset, size);
        int kFold = 5;

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, false);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews
//        analyzer.saveCVIndex(kFold, cvIndexFile);

        // we store the interaction information before-hand, load them directly
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.loadInteractions(friendFile);

        /***save user id file and network file****/
//        analyzer.saveUserIds(userIdFile);
//        analyzer.loadInteractions(friendFile);
//        analyzer.saveNetwork(friendFile);

        /***save files for running baselines***/
//        analyzer.saveAdjacencyMatrix("./data/YelpNewAdjMatrix_10k.txt");
//        analyzer.printDocs4Plane("./data/YelpNew4PlaneDocs_1000.txt");
//        analyzer.printNetwork4Plane("./data/YelpNew4PlaneNetwork_1000.txt");

//        analyzer.loadCVIndex(cvIndexFile);

        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 100, number_of_topics = 10, varMaxIter = 20, embeddingDim = 10;
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();
        EUB eub = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        eub.buildLookupTables(analyzer.getUsers());
        eub.EMonCorpus();

        eub.printTopWords(30, "./data/topkWords.txt");
        eub.printTopicEmbedding("./data/topicEmbedding.txt");
        eub.printUserEmbedding("./data/userEmbedding.txt");

//        eub.fixedCrossValidation(kFold);

        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        System.out.println("[Info]End time: " + end);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60);
        System.out.print(String.format("[Time]This training+testing process took %.2f mins.\n", hours));

    }
}
