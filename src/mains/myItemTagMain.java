package mains;

import Analyzer.BipartiteAnalyzer;
import Analyzer.MultiThreadedReviewAnalyzer;
import Application.ItemTagging;
import structures._Corpus;
import structures._Doc;
import structures._Review;

import java.io.*;
import java.text.ParseException;

/***
 * @author Lu Lin
 */
public class myItemTagMain {
    public static void main(String[] args) throws IOException, ParseException {
        int classNumber = 5;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        double trainRatio = 0, adaptRatio = 1;
        int crossV = 2;
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        boolean enforceAdapt = true;
        String tokenModel = "./data/Model/en-token.bin"; // Token model.
        String fs = "DF";//"IG_CHI"
        int lmTopK = 1000; // topK for language model.
        String lmFvFile = null;

        /***parameter setting***/

        /***data setting***/
        String trainset = "byUser_4k_review";
        String source = "yelp";
        String dataset = "./myData/" + source + "/" + trainset + "/";
        String outputFolder = dataset + "output/" + crossV + "foldsCV" + "/";
        String model = "LDA_Variational";
        String mode = "Embed";
        int number_of_topics = 20;

        String[] fvFiles = new String[4];
        fvFiles[0] = "./data/Features/fv_2gram_IG_yelp_byUser_30_50_25.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG_amazon_movie_byUser_40_50_12.txt";
        fvFiles[2] = "./data/Features/fv_2gram_IG_amazon_electronic_byUser_20_20_5.txt";
        fvFiles[3] = "./data/Features/fv_2gram_IG_amazon_book_byUser_40_50_12.txt";
        int fvFile_point = 0;
        if (source.equals("amazon_movie")) {
            fvFile_point = 1;
        } else if (source.equals("amazon_electronic")) {
            fvFile_point = 2;
        } else if (source.equals("amazon_book")) {
            fvFile_point = 3;
        }

        String reviewFolder = dataset + crossV + "foldsCV/";
        ItemTagging analyzer = new ItemTagging(tokenModel, classNumber, fvFiles[fvFile_point],
                Ngram, lengthThreshold, numberOfCores, true, source);
        analyzer.setMode(mode);
        analyzer.setModel(model);
        analyzer.setTopK(10);
        System.out.println("[Info]Start FIXED cross validation...");
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
            analyzer.loadItemWeight(String.format("%s%d/%s_postByItem_%d.txt", outputFolder, k, model, number_of_topics));
            analyzer.buildItemProfile();
            analyzer.constructTagSet("./myData/" + source + "/business.json");//construct tagset
            analyzer.loadModel(String.format("%s%d/%s_beta_%d.txt", outputFolder, k, model, number_of_topics));
            analyzer.calculateTagging(String.format("%s%d/ItemTag/", outputFolder, k));
        }
    }
}
