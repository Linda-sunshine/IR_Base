package mains;

import Analyzer.MultiThreadedLMAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithETBIR;
import structures.TopicModelParameter;
import structures._User;
import utils.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;

public class ETBIRCFExecution {
    public static void main(String[] args) throws IOException, ParseException {
        TopicModelParameter param = new TopicModelParameter(args);

        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        int lengthThreshold = 5; //Document length threshold
        double trainRatio = 0, adaptRatio = 1;
        boolean enforceAdapt = true;
        boolean setRandomFold = false;
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        String lmFvFile = null;

        String tokenModel = "./data/Model/en-token.bin";
        String folder = String.format("%s/%s/%s", param.m_prefix, param.m_source, param.m_set);
        String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);

        String inputFolder = String.format("%s/%dfoldsCV%s", folder,  param.m_crossV, param.m_flag_coldstart?"Coldstart":"");
        String outputFolder = String.format("%s/output/%dfoldsCV%s", folder,  param.m_crossV, param.m_flag_coldstart?"Coldstart":"");

        /*****experiment setting*****/
        int[] neighborK = new int[]{2,4,6}; // top_k neighbors
        int[] threshold = new int[]{Integer.valueOf(args[0])}; // popularity of item or time
//        int[] topicNums = new int[]{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};// number of topics
        int[] topicNums = new int[]{5, 15, 20, 25, 30, 35, 40, 45, 50};// number of topics
        String[] models = new String[]{args[1]};
        /***meaning of different modes:
         columnPhi: compare phi of document of the same item across different users (rowPhi: same user across different items)
         columnPost: compare posterior parameter (\gamma or softmax(\mu)) of the same item across different users (rowPost: same user across different items)
         columnProduct: compare the inner product of user's P and item's \eta of the same item across different users (rowProduct: same user across different items)
         userEmbded: P for ETBIR, average over documents across users for LDA (\gamma) and CTM (softmax(\mu))
         itemEmbed: \eta for ETBIR, average over documents across items for LDA (\gamma) and CTM (softmax(\mu)) */
        String[] modes = new String[]{"rowPost", "columnPost", "userEmbed", "itemEmbed", "rowProduct", "columnProduct"};

        double[] ndcg, map;
        int dim;
        String mode, model;
        String suffix1 = "txt", suffix2 = "classifer";
        String neighborSelection = "all"; // "all"

        MultiThreadedLMAnalyzer analyzer;
        for(int th : threshold) {
            for(int nk : neighborK) {
                for (int m = 0; m < modes.length; m++) {
                    mode = modes[m];
                    for (int d = 0; d < models.length; d++) {
                        model = models[d];
                        System.out.format("===== %d threshould, %d neighbors, %s mode, %s model =====\n", th, nk, mode, model);
                        for (int n = 0; n < topicNums.length; n++) {
                            dim = topicNums[n];
                            if ((mode.equals("columnProduct") || mode.equals("rowProduct")) && (!model.equals("ETBIR"))) {
                                System.err.format("[Warning]%s mode is only for ETBIR, not for %s.\n", mode, model);
                                continue;
                            }

                            String resultFile = String.format("%s/CF/CF_%s_%d_%d_%s_%s_%d.txt", outputFolder, param.m_source, th, nk, model, mode, dim);
                            File file = new File(resultFile);
                            try {
                                file.getParentFile().mkdirs();
                                file.createNewFile();
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                            try {
                                PrintWriter resultWriter = new PrintWriter(file);

                                ndcg = new double[param.m_crossV];
                                map = new double[param.m_crossV];
                                for (int i = 0; i < param.m_crossV; i++) {
                                    /***Loading data.***/
                                    analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, fvFile,
                                            lmFvFile, Ngram, lengthThreshold, numberOfCores, true);
                                    analyzer.config(trainRatio, adaptRatio, enforceAdapt);

                                    String trainFolder, testFolder = String.format("%s/%d/", inputFolder, i);
                                    //load train set
                                    for (int j = 0; j < param.m_crossV; j++) {
                                        if (j != i) {
                                            trainFolder = String.format("%s/%d/", inputFolder, j);
                                            analyzer.loadUserDir(trainFolder);
                                        }
                                    }
                                    //need to explicitly allocate train-test for each user
                                    for (_User u : analyzer.getUsers()) {
                                        u.constructTrainTestReviews();
                                    }
                                    //load test set
                                    analyzer.loadTestUserDir(testFolder);
                                    analyzer.setFeatureValues("TFIDF-sublinear", 0);

                                    /***Collaborative filtering starts here.***/
                                    boolean equalWeight = false;
                                    String saveAdjFolder = String.format("%s/%d/", outputFolder, i);
                                    String dir, cfFile;


                                    /***
                                     * In order to perform cf, we need to follow the following steps:
                                     * Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
                                     * Step 2: perform collaborative filtering */

                                    // Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
                                    CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize() + 1);
                                    // construct ranking neighbors
                                    dir = String.format("%s/%s_cf_%s_%d_", saveAdjFolder, param.m_source, neighborSelection, th);
                                    cfInit.constructRankingNeighbors(neighborSelection, th);

                                    // Step 2: perform collaborative filtering
                                    // load candidate items from the saved neighbor file
//                                    cfInit.saveUserItemPairs(dir);
//                                    cfFile = String.format("%s/%s_cf_%s_%d_test.csv", saveAdjFolder, param.m_source, neighborSelection, th);
//                                    cfInit.loadRankingCandidates(cfFile);
                                    cfInit.constructRankingCandidates();

                                    ArrayList<_User> cfUsers = cfInit.getUsers();
                                    int validUser = cfInit.getValidUserSize();
                                    System.out.format("\n-----------------run %s model for %s mode-------------------------\n", model, mode);
                                    CollaborativeFilteringWithETBIR cf = new CollaborativeFilteringWithETBIR(cfUsers, analyzer.getFeatureSize() + 1, nk, dim);
                                    cf.setValidUserSize(validUser);
                                    cf.setEqualWeightFlag(equalWeight);
                                    cf.setMode(mode);
                                    cf.setModel(model);

                                    String userWeight = "", itemWeight = "", docWeight = "";
                                    if (mode.equals("rowPhi") || mode.equals("columnPhi")) {
                                        userWeight = String.format("%s/%d/%s_phiByUser_%d.txt", outputFolder, i, model, dim);//user embedding
                                        itemWeight = String.format("%s/%d/%s_phiByItem_%d.txt", outputFolder, i, model, dim);//item embedding
                                        docWeight = String.format("%s/%d/%s_phi_%d.txt", outputFolder, i, model, dim);//doc embedding
                                    } else if (mode.equals("rowPost") || mode.equals("columnPost")) {
                                        userWeight = String.format("%s/%d/%s_postByUser_%d.txt", outputFolder, i, model, dim);//user embedding
                                        itemWeight = String.format("%s/%d/%s_postByItem_%d.txt", outputFolder, i, model, dim);//item embedding
                                        if (model.equals("LDA_Variational"))
                                            docWeight = String.format("%s/%d/%s_postGamma_%d.txt", outputFolder, i, model, dim);//doc embedding
                                        else
                                            docWeight = String.format("%s/%d/%s_postSoftmax_%d.txt", outputFolder, i, model, dim);//doc embedding
                                    } else if (mode.equals("userEmbed") || mode.equals("itemEmbed")) {
                                        if (model.equals("ETBIR")) {
                                            userWeight = String.format("%s/%d/%s_postNu_%d.txt", outputFolder, i, model, dim);//user embedding
                                            itemWeight = String.format("%s/%d/%s_postEta_%d.txt", outputFolder, i, model, dim);//item embedding
                                            docWeight = String.format("%s/%d/%s_phi_%d.txt", outputFolder, i, model, dim);//doc embedding, here won't be used
                                        } else {
                                            userWeight = String.format("%s/%d/%s_postByUser_%d.txt", outputFolder, i, model, dim);//user embedding
                                            itemWeight = String.format("%s/%d/%s_postByItem_%d.txt", outputFolder, i, model, dim);//item embedding
                                            docWeight = String.format("%s/%d/%s_phi_%d.txt", outputFolder, i, model, dim);//doc embedding, here won't be used
                                        }
                                    } else if (mode.equals("rowProduct") || mode.equals("columnProduct")) {
                                        userWeight = String.format("%s/%d/%s_postNu_%d.txt", outputFolder, i, model, dim);//user embedding
                                        itemWeight = String.format("%s/%d/%s_postEta_%d.txt", outputFolder, i, model, dim);//item embedding
                                        docWeight = String.format("%s/%d/%s_phi_%d.txt", outputFolder, i, model, dim);//doc embedding, here won't be used
                                    }

                                    cf.loadWeights(userWeight, model, suffix1, suffix2);
                                    cf.loadItemWeights(itemWeight, model);
                                    cf.loadReviewTopicWeights(docWeight);

                                    cf.calculateAllNDCGMAP();
                                    cf.calculateAvgNDCGMAP();

                                    System.out.format("\n[Info]NDCG: %.4f, MAP: %.4f\n", cf.getAvgNDCG(), cf.getAvgMAP());
                                    ndcg[i] = cf.getAvgNDCG();
                                    map[i] = cf.getAvgMAP();
                                }

                                //printout the performance statistics
                                double mean1 = Utils.sumOfArray(ndcg) / param.m_crossV, var1 = 0;
                                for (int i = 0; i < ndcg.length; i++)
                                    var1 += (ndcg[i] - mean1) * (ndcg[i] - mean1);
                                var1 = Math.sqrt(var1 / param.m_crossV);
                                System.out.format("[Stat-NDCG]%d threshold, %d neighbors, %d topic number, %s mode, %s model: %.4f+/-%.4f\n",
                                        th, nk, dim, mode, model, mean1, var1);

                                double mean2 = Utils.sumOfArray(map) / param.m_crossV, var2 = 0;
                                for (int i = 0; i < map.length; i++)
                                    var2 += (map[i] - mean2) * (map[i] - mean2);
                                var2 = Math.sqrt(var2 / param.m_crossV);
                                System.out.format("[Stat-MAP]%d threshold, %d neighbors, %d topic number, %s mode, %s model: %.4f+/-%.4f\n",
                                        th, nk, dim, mode, model, mean2, var2);

                                resultWriter.format("[Stat]NDCG %.5f+/-%.5f\n", mean1, var1);
                                resultWriter.format("[Stat]MAP %.5f+/-%.5f\n", mean2, var2);
                                resultWriter.close();
                            } catch (FileNotFoundException ex) {
                                System.err.format("[Error]Failed to open file %s\n", resultFile);
                            }
                        }
                    }
                }
            }
        }
    }
}
