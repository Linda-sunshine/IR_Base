package mains;

import Analyzer.MultiThreadedReviewAnalyzer;
import Application.ItemTagging;
import structures.TopicModelParameter;
import structures._Doc;
import structures._Review;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;

public class myETBIRItemTagExcecution {
    public static void main(String[] args) throws IOException, ParseException {
        TopicModelParameter param = new TopicModelParameter(args);

        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        int lengthThreshold = 5; //Document length threshold
        boolean setRandomFold = false;
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String tokenModel = "./data/Model/en-token.bin";
        String dataset = String.format("%s/%s/%s", param.m_prefix, param.m_source, param.m_set);
        String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);
        String reviewFolder = String.format("%s/%dfoldsCV%s/", dataset, param.m_crossV, param.m_flag_coldstart?"Coldstart":"");
        String outputFolder = String.format("%s/output/%dfoldsCV%s/", dataset, param.m_crossV, param.m_flag_coldstart?"Coldstart":"");

        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFile,
                Ngram, lengthThreshold, numberOfCores, true, param.m_source);
//        analyzer.setReleaseContent(false);//Remember to set it as false when generating crossfolders!!!
//        analyzer.loadUserDir(reviewFolder);

        ItemTagging tagger = new ItemTagging(tokenModel, classNumber, fvFile,
                Ngram, lengthThreshold, numberOfCores, true, param.m_source);
        tagger.setMode(param.m_mode);
        tagger.setModel(param.m_topicmodel);
        tagger.setTopK(10);

        System.out.println("[Info]Start FIXED cross validation...");
        //first load corpus, aka tags
        tagger.loadCorpus(String.format("%s/%s/business.json",param.m_prefix, param.m_source));
        double[] map = new double[param.m_crossV];
        double[] precision = new double[param.m_crossV];

        long starttime = System.currentTimeMillis();
        for (int k = 0; k < param.m_crossV; k++) {
            analyzer.getCorpus().reset();
            //load test set
            String testFolder = reviewFolder + k + "/";
            analyzer.loadUserDir(testFolder);
            for (_Doc d : analyzer.getCorpus().getCollection()) {
                d.setType(_Review.rType.TEST);
            }
            //load train set
            for (int i = 0; i < param.m_crossV; i++) {
                if (i != k) {
                    String trainFolder = reviewFolder + i + "/";
                    analyzer.loadUserDir(trainFolder);
                }
            }

            String modelFile = String.format("%s/%d/%s_beta_%d.txt", outputFolder, k, param.m_topicmodel, param.m_number_of_topics);
            String itemWeightFile;
            if(param.m_topicmodel.equals("ETBIR") || param.m_topicmodel.equals("ETBIR_User") || param.m_topicmodel.equals("ETBIR_Item"))
                itemWeightFile = String.format("%s/%d/%s_postEta_%d.txt", outputFolder, k, param.m_topicmodel, param.m_number_of_topics);
            else
                itemWeightFile = String.format("%s/%d/%s_postByItem_%d.txt", outputFolder, k, param.m_topicmodel, param.m_number_of_topics);
            tagger.loadItemWeight(itemWeightFile, 0);

            tagger.buildItemProfile(analyzer.getCorpus().getCollection());
            tagger.loadModel(modelFile);
            double[] results = tagger.calculateTagging(String.format("%s/%d/ItemTag/", outputFolder, k));
            map[k] = results[0];
            precision[k] = results[1];
        }
        System.out.println();

        long endtime = System.currentTimeMillis();
        NumberFormat formatter = new DecimalFormat("#0.00000");
        System.out.format("[Stat]Running time (seconds) %s\n", formatter.format((endtime-starttime) / 1000d));

        double mean = 0, var = 0;
        for (int i = 0; i < map.length; i++) {
            mean += map[i];
        }
        mean /= map.length;
        for (int i = 0; i < map.length; i++) {
            var += (map[i] - mean) * (map[i] - mean);
        }
        var = Math.sqrt(var / map.length);
        System.out.format("[Stat]MAP %.3f+/-%.3f\n", mean, var);

        mean = 0;
        var = 0;
        for (int i = 0; i < precision.length; i++) {
            mean += precision[i];
        }
        mean /= map.length;
        for (int i = 0; i < precision.length; i++) {
            var += (precision[i] - mean) * (precision[i] - mean);
        }
        var = Math.sqrt(var / precision.length);
        System.out.format("[Stat]Precision %.3f+/-%.3f\n", mean, var);
    }
}
