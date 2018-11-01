package myMains;
import Analyzer.MultiThreadedNetworkAnalyzer;
import opennlp.tools.util.InvalidFormatException;

import java.io.FileNotFoundException;
import java.io.IOException;


/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The main entrance for calling EUB model
 */
public class MyEUBProcesMain {

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

        for(int k: new int[]{0, 1, 2, 3, 4}) {
            int kFold = 5;
            int time = 2;

            String orgFriendFile = String.format("%s/%s/%sFriends_org.txt", prefix, dataset, dataset);
            String friendFile = String.format("%s/%s/%sFriends.txt", prefix, dataset, dataset);
            String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", prefix, dataset, dataset);
            String cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction.txt", prefix, dataset, dataset);
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


            /***Date processing for related experiments of EUB****/
            analyzer.loadUserDir(userFolder);
            analyzer.constructUserIDIndex();
            analyzer.loadInteractions(friendFile);

            int d1 = 10, d2 = 50, e1 = 5, e2 = 20;
            if (dataset.equals("StackOverflow")) {
                d1 = 15; d2 = 50; e1 = 5; e2 = 15;
            }

//        analyzer.calcDocStat4LightMediumHeavy(d1, d2);
//        analyzer.calcEdgeStat4LightMediumHeavy(e1, e2);

            // cold start of docs -- split based on connectivity
            int sampleSize = 200;
//            String coldStart4DocsFile = String.format("./data/CoLinAdapt/%s/%s_cold_start_4docs_fold_%d.txt", dataset, dataset, k);
//            analyzer.sampleUsers4ColdStart4Docs(coldStart4DocsFile, e1, e2, sampleSize);

            // cold start of edges -- split based on document size
            String coldStart4EdgesFile = String.format("./data/CoLinAdapt/%s/%s_cold_start_4edges_fold_%d", dataset, dataset, k);
            analyzer.sampleUsers4ColdStart4Edges(coldStart4EdgesFile, d1, d2, sampleSize);

        }
//        String mode = "cv4doc"; // "cv4edge"
//
//        //if it is cv for doc, use all the interactions + part of docs
//        if(mode.equals("cv4doc")){
//            analyzer.loadInteractions(friendFile);
//            analyzer.loadCVIndex(cvIndexFile, kFold);
//            // if it is cv for edge, use all the docs + part of edges
//        } else if(mode.equals("cv4edge")){
//            // if no cv index file for doc is loaded, then all documents will be training docs.
//            cvIndexFile4Interaction = String.format("%s/%s/%sCVIndex4Interaction_fold_%d_train.txt", prefix, dataset, dataset, k);
//            analyzer.loadInteractions(cvIndexFile4Interaction);
//        }
    }
}
