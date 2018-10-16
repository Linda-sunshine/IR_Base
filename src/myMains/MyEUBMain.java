package myMains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Analyzer.UserAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import topicmodels.UserEmbedding.EUB;

import java.io.File;
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

        String dataset = "StackOverflow"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        boolean server = false;
        boolean cvFlag = dataset.equals("StackOverflow") ? true : false;

		String prefix = server ? "/zf8/lg5bt/DataSigir" : "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/SelectedVocab.csv", prefix, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);

        /***Feature selection**/
//        UserAnalyzer analyzer = new UserAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, cvFlag);
//        analyzer.loadUserDir(userFolder);
//        analyzer.featureSelection("./data/StackOverflowSelectedVocab_DF_8k.txt", "DF",8000, 100, 5000);

        String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
        String userIdFile = String.format("%s/%s/%sUserIds.txt", prefix, dataset, dataset);

        int kFold = 5;
        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, cvFlag);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        // we store the interaction information before-hand, load them directly
//        analyzer.loadUserDir(userFolder);
//        analyzer.constructUserIDIndex();
//        analyzer.saveCVIndex(kFold, cvIndexFile);
//        analyzer.loadInteractions(friendFile);
//        analyzer.saveNetwork(friendFile);

        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.loadInteractions(friendFile);
        analyzer.loadCVIndex(cvIndexFile);

        /***save user id file and network file****/
//        analyzer.saveUserIds(userIdFile);
//        analyzer.loadInteractions(friendFile);
//        analyzer.saveNetwork(friendFile);

        /***save files for running baselines***/
//        analyzer.saveAdjacencyMatrix("./data/YelpNewAdjMatrix_10k.txt");
//        analyzer.printDocs4Plane("./data/YelpNew4PlaneDocs_1000.txt");
//        analyzer.printNetwork4Plane("./data/YelpNew4PlaneNetwork_1000.txt");


        /***Start running joint modeling of user embedding and topic embedding****/
        int emMaxIter = 10, number_of_topics = 10, varMaxIter = 20, embeddingDim = 10;
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();

        EUB eub = new EUB(emMaxIter, emConverge, beta, corpus, lambda, number_of_topics, alpha, varMaxIter, varConverge, embeddingDim);
        eub.buildLookupTables(analyzer.getUsers());
        eub.EMonCorpus();
        eub.setDisplayLv(0);
//        eub.fixedCrossValidation(kFold);
        long end = System.currentTimeMillis();

        // record related information
        String savePrefix = server ? "/zf8/lg5bt/embedExp/eub" : "./data/embedExp/eub";
        String saveDir = String.format("%s/%d_%s", savePrefix, start, dataset);
        File fileDir = new File(saveDir);
        if(!fileDir.exists())
            fileDir.mkdirs();

        eub.printTopWords(30, String.format("%s/topkWords.txt", saveDir));
        eub.printTopicEmbedding(String.format("%s/topicEmbedding.txt", saveDir));
        eub.printUserEmbedding(String.format("%s/userEmbedding.txt", saveDir));

        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60);
//        System.out.print(String.format("[Time]This training+testing process took %.2f mins.\n", hours));

    }
}
