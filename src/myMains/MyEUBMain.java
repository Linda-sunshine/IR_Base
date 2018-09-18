package myMains;

import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Classifier.supervised.modelAdaptation.MMB.MTCLinAdaptWithMMB;
import opennlp.tools.util.InvalidFormatException;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

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
        MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, false);
        analyzer.loadUserDir(userFolder);
        analyzer.buildFriendship(friendFile);

        analyzer.setFeatureValues("TFIDF-sublinear", 0);
        HashMap<String, Integer> featureMap = analyzer.getFeatureMap();


        MTCLinAdaptWithMMB mmb = new MTCLinAdaptWithMMB(classNumber, analyzer.getFeatureSize(), featureMap, globalModel, featureGroupFile, featureGroupFileSup, globalLM);
        mmb.setR2TradeOffs(eta3, eta4);

        mmb.setsdA(sdA);
        mmb.setsdB(sdB);

        mmb.setR1TradeOffs(eta1, eta2);
        mmb.setConcentrationParams(alpha, eta, beta);

        double rho = 0.2;
        int burnin = 10, iter = 30, thin = 3;
        boolean jointAll = false;
        mmb.setRho(rho);
        mmb.setBurnIn(burnin);
        mmb.setThinning(thin);// default 3
        mmb.setNumberOfIterations(iter);

        mmb.setJointSampling(jointAll);
        mmb.loadLMFeatures(analyzer.getLMFeatures());
        mmb.loadUsers(analyzer.getUsers());
        mmb.setDisplayLv(displayLv);
        long start = System.currentTimeMillis();

        boolean trace = true;
        if(trace){
            iter = 200; thin = 1; burnin = 0;
            mmb.setNumberOfIterations(iter);
            mmb.setThinning(thin);
            mmb.setBurnIn(burnin);
            mmb.trainTrace(dataset, start);
        } else{
            mmb.train();
            mmb.test();
        }
        long end = System.currentTimeMillis();
        System.out.println("\n[Info]Start time: " + start);
        System.out.println("[Info]End time: " + end);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60);
        System.out.print(String.format("[Time]This training+testing process took %.2f mins.\n", hours));

    }
}
