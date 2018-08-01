package mains;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedReviewAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithETBIR;
import utils.Utils;

public class ETBIRCFMain {
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

        int classNumber = 5;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        double trainRatio = 0, adaptRatio = 1;
        int[] ks = new int[]{2, 4, 6, 8, 10, 12, 14}; // top_k neighbors
        int crossV = 5;
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        boolean enforceAdapt = true;
        String tokenModel = "./data/Model/en-token.bin"; // Token model.
        String fs = "DF";//"IG_CHI"
        int lmTopK = 1000; // topK for language model.
        String lmFvFile = null;
//        String lmFvFile = String.format("./data/CoLinAdapt/%s/fv_lm_%s_%d.txt", dataset, fs, lmTopK);

        /*****data setting*****/
        String scaleset = "byUser_4k_review";
        String dataset = "yelp";
        String folder = String.format("./myData/%s/%s", dataset, scaleset);
        String inputFolder = String.format("%s/%dfoldsCV", folder, crossV);
        String outputFolder = String.format("%s/output/%dfoldsCV", folder, crossV);

        String[] fvFiles = new String[4];
        fvFiles[0] = "./data/Features/fv_2gram_IG_yelp_byUser_30_50_25.txt";
        fvFiles[1] = "./data/Features/fv_2gram_IG_amazon_movie_byUser_40_50_12.txt";
        fvFiles[2] = "./data/Features/fv_2gram_IG_amazon_electronic_byUser_20_20_5.txt";
        fvFiles[3] = "./data/Features/fv_2gram_IG_amazon_book_byUser_40_50_12.txt";
        int fvFile_point = 0;
        if(dataset.equals("amazon_movie")){
            fvFile_point = 1;
        }else if(dataset.equals("amazon_electronic")){
            fvFile_point = 2;
        }else if(dataset.equals("amazon_book")){
            fvFile_point = 3;
        }

        double[][] ndcg = new double[ks.length][crossV];
        double[][] map = new double[ks.length][crossV];
        for(int i = 0; i < crossV; i++) {
            /***Loading data.***/
            MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, fvFiles[fvFile_point],
                    lmFvFile, Ngram, lengthThreshold, numberOfCores, true);
            analyzer.config(trainRatio, adaptRatio, enforceAdapt);

            String trainFolder, testFolder = String.format("%s/%d/", inputFolder, i);
            //load train set
            for(int j = 0; j < crossV; j++){
                if(j != i){
                    trainFolder = String.format("%s/%d/", inputFolder, j);
                    analyzer.loadUserDir(trainFolder);
                }
            }
            //need to explicitly allocate train-test for each user
            for(_User u:analyzer.getUsers()){
                u.constructTrainTestReviews();
            }
            //load test set
            analyzer.loadTestUserDir(testFolder);
            analyzer.setFeatureValues("TFIDF-sublinear", 0);

            /***Collaborative filtering starts here.***/
            boolean equalWeight = false;
            String saveAdjFolder = String.format("%s/%d/", outputFolder, i);
            String dir, cfFile, model;
            String[] models = new String[]{"ETBIR","LDA","CTM"};
            String mode = "userP"; //row, column, userP, itemEta, productUserItem
            String suffix1 = "txt", suffix2 = "classifer";
            String neighborSelection = "all"; // "all"

            /***
             * In order to perform cf, we need to follow the following steps:
             * Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
             * Step 2: perform collaborative filtering */

            // Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
            CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize() + 1);
            int dim = 20;
            int[] threshold = new int[]{3, 5, 7, 10};

            for (int th : threshold) {
                dir = String.format("%s/%s_cf_%s_%d_", saveAdjFolder, dataset, neighborSelection, th);
                // construct ranking neighbors
                cfInit.constructRankingNeighbors(neighborSelection, th);
                cfInit.saveUserItemPairs(dir);
            }

            // Step 2: perform collaborative filtering
            for (int th : threshold) { // threshold: time or popularity
                for (int k = 0; k < ks.length; k++) {
                    // load the saved neighbor file
                    cfFile = String.format("%s/%s_cf_%s_%d_test.csv", saveAdjFolder, dataset, neighborSelection, th);
                    ArrayList<_User> cfUsers = cfInit.getUsers();
                    cfInit.loadRankingCandidates(cfFile);

                    int validUser = cfInit.getValidUserSize();

                    for(int m = 0; m < models.length; m++) {
                        model = models[m];
                        System.out.format("\n-----------------run %s %d neighbors-------------------------\n", model, ks[k]);
                        CollaborativeFilteringWithETBIR cf = new CollaborativeFilteringWithETBIR(cfUsers, analyzer.getFeatureSize() + 1, ks[k], dim);
                        cf.setValidUserSize(validUser);
                        cf.setEqualWeightFlag(equalWeight);
                        cf.setMode(mode);

                        if(model.equals("ETBIR")) {
                            String userWeight = String.format("%s/%d/%s_postNu.txt", outputFolder, i, model);//user embedding
                            String itemWeight = String.format("%s/%d/%s_postEta.txt", outputFolder, i, model);//item embedding
                            cf.loadWeights(userWeight, model, suffix1, suffix2);
                            cf.loadItemWeights(itemWeight);
                        } else if(model.equals("LDA")){
                            String userWeight = String.format("%s/%d/%s_postByUser.txt", outputFolder, i, model);
                            String itemWeight = String.format("%s/%d/%s_postByItem.txt", outputFolder, i, model);
                        }

                        cf.calculateAllNDCGMAP();
                        cf.calculateAvgNDCGMAP();

                        System.out.format("\n[Info]NDCG: %.4f, MAP: %.4f\n", cf.getAvgNDCG(), cf.getAvgMAP());
                        ndcg[k][i] = cf.getAvgNDCG();
                        map[k][i] = cf.getAvgMAP();
                    }
                }
            }
        }

        //output the performance statistics
        for(int k = 0; k < ks.length; k++) {
            double mean = Utils.sumOfArray(ndcg[k]) / crossV, var = 0;
            for (int i = 0; i < ndcg[k].length; i++)
                var += (ndcg[k][i] - mean) * (ndcg[k][i] - mean);
            var = Math.sqrt(var / crossV);
            System.out.format("[Stat]NDCG %.4f+/-%.4f\n", mean, var);

            mean = Utils.sumOfArray(map[k]) / crossV;
            var = 0;
            for (int i = 0; i < map[k].length; i++)
                var += (map[k][i] - mean) * (map[k][i] - mean);
            var = Math.sqrt(var / crossV);
            System.out.format("[Stat]MAP %.4f+/-%.4f\n", mean, var);
        }
    }
}
