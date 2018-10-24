package myMains;

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

        boolean cvFlag = dataset.equals("StackOverflow") ? true : false;

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        /***Feature selection**/
//        UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, cvFlag);
//        analyzer.loadUserDir(userFolder);
//        analyzer.featureSelection("./data/StackOverflowSelectedVocab_DF_8k.txt", "DF",8000, 100, 5000);

//        int time = 2;
        for(int time: new int[]{2, 3, 4, 5, 6, 7, 8}) {
            String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
            String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
            String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
            String cvIndexFile4NonInteraction = String.format("%s/%s/%sCVIndex4NonInteraction-%d.txt", prefix, dataset, dataset, time);

            int kFold = 5, k = 0;
            MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                    Ngram, lengthThreshold, numberOfCores, cvFlag);
            analyzer.setAllocateReviewFlag(false); // do not allocate reviews

//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.printData4TADW("./data/StackOverflowTADW.txt");

            /****we store the interaction information and cv index before-hand****/
            analyzer.loadUserDir(userFolder);
            analyzer.constructUserIDIndex();
            //analyzer.saveCVIndex(5, cvIndexFile);

            analyzer.loadInteractions(friendFile);
            analyzer.assignCVIndex4Network(kFold, time);
            analyzer.sanityCheck4CVIndex4Network(true);
            analyzer.sanityCheck4CVIndex4Network(false);

            if (time == 2)
                analyzer.saveCVIndex4Network(cvIndexFile4Interaction, true);
            analyzer.saveCVIndex4Network(cvIndexFile4NonInteraction, false);
        }
        /***save user id file and network file****/
//        analyzer.saveUserIds(userIdFile);
//        analyzer.loadInteractions(friendFile);
//        analyzer.saveNetwork(friendFile);

        /***save files for running baselines***/
//        analyzer.saveAdjacencyMatrix("./data/YelpNewAdjMatrix_10k.txt");
//        analyzer.printDocs4Plane("./data/YelpNew4PlaneDocs_1000.txt");
//        analyzer.printNetwork4Plane("./data/YelpNew4PlaneNetwork_1000.txt");

//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.loadInteractions(friendFile);
//        analyzer.loadCVIndex(cvIndexFile, kFold);
//
//        /***Start running joint modeling of user embedding and topic embedding****/
//        int emMaxIter = 50, number_of_topics = 20, varMaxIter = 1, embeddingDim = 10;
//        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
//        _Corpus corpus = analyzer.getCorpus();
//
//        // start the vi process
//        long start = System.currentTimeMillis();
//        LDA_Variational tModel = null;
//        int innerIter = 1;
//        boolean multiFlag = true, adaFlag = false;
//        if(multiFlag)
//            tModel = new EUB_multithreading(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
//        else
//            tModel = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
//
//        boolean alphaFlag = false, gammaFlag = false, betaFlag = false, tauFlag = false, xiFlag = false;
//        ((EUB) tModel).setModelParamsUpdateFlags(alphaFlag, gammaFlag, betaFlag, tauFlag, xiFlag);
//        ((EUB) tModel).initLookupTables(analyzer.getUsers());
//        ((EUB) tModel).setDisplayLv(0);
//        ((EUB) tModel).setInnerMaxIter(innerIter);
//
//        long current = System.currentTimeMillis();
//        String saveDir = String.format("./data/emebddingExp/eub/%s_emIter_%d_nuTopics_%d_varIter_%d_innerIter_%d_dim_%d_ada_%b/" +
//                "fold_%d_%d", dataset, emMaxIter, number_of_topics, varMaxIter, innerIter, embeddingDim, adaFlag, kFold, current);
//        ((EUB) tModel).fixedCrossValidation(k, saveDir);
//        long end = System.currentTimeMillis();
//
//        System.out.println("\n[Info]Start time: " + start);
//        // the total time of training and testing in the unit of hours
//        double hours = (end - start)/(1000*60*60);
//        System.out.print(String.format("[Time]This training+testing process took %.4f hours.\n", hours));

    }
}
