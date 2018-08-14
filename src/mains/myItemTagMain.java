package mains;

import Analyzer.BipartiteAnalyzer;
import Analyzer.MultiThreadedReviewAnalyzer;
import Application.ItemTagging;
import structures._Corpus;
import structures._Doc;
import structures._Review;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.ParseException;

/***
 * @author Lu Lin
 */
public class myItemTagMain {
    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        String tokenModel = "./data/Model/en-token.bin";

        /***parameter setting***/

        /***data setting***/
        int crossV = 2;
        String trainset = "byUser_4k_review";
        String source = "yelp";
        String dataset = "./myData/" + source + "/" + trainset + "/";
        String outputFolder = dataset + "output/" + crossV + "foldsCV" + "/";
        String model = "ETBIR";
        String mode = "Embed";
        int number_of_topics = 20;

        String[] fvFiles = new String[4];
        fvFiles[0] = "./data/Features/fv_2gram_IG_yelp_byUser_30_50_25.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG_amazon_movie_byUser_40_50_12.txt";
        fvFiles[2] = "./data/Features/fv_2gram_IG_amazon_electronic_byUser_20_20_5.txt";
        fvFiles[3] = "./data/Features/fv_2gram_IG_amazon_book_byUser_40_50_12.txt";
        int fvFile_point = 0;
        if(source.equals("amazon_movie")){
            fvFile_point = 1;
        }else if(source.equals("amazon_electronic")){
            fvFile_point = 2;
        }else if(source.equals("amazon_book")){
            fvFile_point = 3;
        }

        String reviewFolder = String.format("%s%dfoldsCV/", dataset, crossV); //2foldsCV/folder0/train/, data/
        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point],
                Ngram, lengthThreshold, numberOfCores, true, source);
//        analyzer.setReleaseContent(false);//Remember to set it as false when generating crossfolders!!!
//        analyzer.loadUserDir(reviewFolder);

        ItemTagging tagger = new ItemTagging(tokenModel, classNumber, fvFiles[fvFile_point],
                Ngram, lengthThreshold, numberOfCores, true, source);
        tagger.setMode(mode);
        tagger.setModel(model);
        tagger.setTopK(10);

        System.out.println("[Info]Start FIXED cross validation...");
        tagger.loadCorpus("./myData/" + source + "/business.json");
        double[] map = new double[crossV];
        double[] precision = new double[crossV];
        double[] mrr = new double[crossV];

        long starttime = System.currentTimeMillis();
        for (int k = 0; k < crossV; k++) {
            analyzer.getCorpus().reset();
            //load test set
            String testFolder = reviewFolder + k + "/";
            analyzer.loadUserDir(testFolder);
            for (_Doc d : analyzer.getCorpus().getCollection()) {
                d.setType(_Review.rType.TEST);
            }
            //load train set
            for (int i = 0; i < crossV; i++) {
                if (i != k) {
                    String trainFolder = reviewFolder + i + "/";
                    analyzer.loadUserDir(trainFolder);
                }
            }
            //first load corpus, aka tags
            tagger.loadItemWeight(String.format("%s%d/%s_postByItem_%d.txt", outputFolder, k, model, number_of_topics), 40);
            tagger.buildItemProfile(analyzer.getCorpus().getCollection());
            tagger.loadModel(String.format("%s%d/%s_beta_%d.txt", outputFolder, k, model, number_of_topics));
            double[] results = tagger.calculateTagging(String.format("%s%d/ItemTag/", outputFolder, k), 0.5);
            map[k] = results[0];
            mrr[k] = results[1];
            precision[k] = results[2];
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
        for (int i = 0; i < mrr.length; i++) {
            mean += mrr[i];
        }
        mean /= mrr.length;
        for (int i = 0; i < mrr.length; i++) {
            var += (mrr[i] - mean) * (mrr[i] - mean);
        }
        var = Math.sqrt(var / mrr.length);
        System.out.format("[Stat]MRR %.3f+/-%.3f\n", mean, var);

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
