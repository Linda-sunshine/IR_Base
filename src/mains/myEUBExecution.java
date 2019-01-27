package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.util.Arrays;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import structures.*;
import topicmodels.CTM.CTM;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Focus_multithread;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.embeddingModel.ETBIR_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import Analyzer.MultiThreadedReviewAnalyzer;
import utils.Utils;

public class myEUBExecution {

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
        String reviewFolder = String.format("%s/data/", dataset);
        String cvIndexFile = String.format("%s/%sCVIndex.txt", dataset, param.m_source);
//        String outputFolder = String.format("%s/output/%s/%s/", param.m_prefix, param.m_source, param.m_set);
        String outputFolder = String.format("%s/output%s/%s/%s/",
                param.m_prefix, param.m_flag_coldstart?"Coldstart":"", param.m_source, param.m_set);

        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, fvFile,
                Ngram, lengthThreshold, numberOfCores, true);
        _Corpus corpus = analyzer.getCorpus();

        int result_dim = 1;
        pLSA tModel = null;
        if (param.m_topicmodel.equals("pLSA")) {
            tModel = new pLSA_multithread(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
                    param.m_lambda, param.m_number_of_topics, param.m_alpha);
        } else if (param.m_topicmodel.equals("LDA_Gibbs")) {
            tModel = new LDA_Gibbs(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
                    param.m_lambda, param.m_number_of_topics, param.m_alpha, 0.4, 50);
        }  else if (param.m_topicmodel.equals("LDA_Variational")) {
            tModel = new LDA_Variational_multithread(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
                    param.m_lambda, param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge);
        }else if (param.m_topicmodel.equals("LDA_User") || param.m_topicmodel.equals("LDA_Item")) {
            tModel = new LDA_Focus_multithread(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
                    param.m_lambda, param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge);
            if(param.m_topicmodel.equals("LDA_User"))
                ((LDA_Focus_multithread) tModel).setMode("User");
            else if(param.m_topicmodel.equals("LDA_Item"))
                ((LDA_Focus_multithread) tModel).setMode("Item");

            result_dim = 5;
        } else if(param.m_topicmodel.equals("ETBIR") || param.m_topicmodel.equals("ETBIR_User") || param.m_topicmodel.equals("ETBIR_Item")){
            tModel = new ETBIR_multithread(param.m_emIter, param.m_emConverge, param.m_beta, corpus, param.m_lambda,
                    param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge, param.m_sigma, param.m_rho);
            if(param.m_topicmodel.equals("ETBIR_User"))
                ((ETBIR_multithread) tModel).setMode("User");
            else if(param.m_topicmodel.equals("ETBIR_Item"))
                ((ETBIR_multithread) tModel).setMode("Item");

            ((ETBIR_multithread) tModel).setFlagGd(param.m_flag_gd);
            ((ETBIR_multithread) tModel).setFlagLambda(param.m_flag_fix_lambda);

            result_dim = 5;
        } else if(param.m_topicmodel.equals("CTM")){
            tModel = new CTM(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
                    param.m_lambda, param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge);
        } else{
            System.out.println("The selected topic model has not developed yet!");
            return;
        }

        tModel.setDisplayLap(1);
        new File(outputFolder).mkdirs();

        if (param.m_crossV<=1) {//just train
            if(param.m_source.equals("StackOverflow2")){
                cvIndexFile = String.format("%s/%sCVIndex4Recommendation.txt", dataset, param.m_source);
                String selectedItemFile = String.format("%s/%sSelectedQuestions.txt", dataset, param.m_source);
                analyzer.setAllocateReviewFlag(false);
                analyzer.loadUserDir(reviewFolder);
                System.out.format("[Dataset]%d docs are loaded.\n", analyzer.getCorpus().getCollection().size());
                analyzer.constructUserIDIndex();
                System.out.format("[Dataset]%d users are loaded.\n", analyzer.getUsers().size());
                analyzer.loadCVIndex(cvIndexFile);
                analyzer.maskDocByCVIndex(0);
                tModel.setCorpus(analyzer.getCorpus());
                System.out.format("[Info]train size = %d....\n", tModel.getTrainSize());

                tModel.EM();
                tModel.printSelectedDocTheta(param.m_topk, outputFolder, param.m_topicmodel, selectedItemFile);
            } else {
                tModel.EMonCorpus();
            }
            tModel.printParameterAggregation(param.m_topk, outputFolder, param.m_topicmodel);
            tModel.printTopWords(param.m_topk);
        } else if(setRandomFold == true){//cross validation with random folds
            analyzer.setAllocateReviewFlag(false);
            reviewFolder = String.format("%s/data/", dataset);
            analyzer.loadUserDir(reviewFolder);
            tModel.setRandomFold(setRandomFold);
            double trainProportion = ((double)param.m_crossV - 1)/(double)param.m_crossV;
            double testProportion = 1-trainProportion;
            tModel.setPerplexityProportion(testProportion);
            tModel.crossValidation(param.m_crossV);
        } else{//cross validation with fixed folds, indexed by CVIndex file
            analyzer.setAllocateReviewFlag(false);
            analyzer.loadUserDir(reviewFolder);
            System.out.format("[Dataset]%d docs are loaded.\n", analyzer.getCorpus().getCollection().size());
            analyzer.constructUserIDIndex();
            System.out.format("[Dataset]%d users are loaded.\n", analyzer.getUsers().size());
            analyzer.loadCVIndex(cvIndexFile);

            if(param.m_flag_coldstart)
                result_dim=4;

            double[][] perf = new double[param.m_crossV][result_dim];
            double[][] like = new double[param.m_crossV][result_dim];
            System.out.println("[Info]Start FIXED cross validation...");
            for(int k = 0; k <param.m_crossV; k++){
                System.out.format("====================\n[Info]%s Fold No. %d: \n",
                        param.m_flag_coldstart?"COLD start":"", k);
                double[] results;

                if(!param.m_flag_coldstart) {
                    analyzer.maskDocByCVIndex(k);
                    tModel.setCorpus(analyzer.getCorpus());
                    results = tModel.oneFoldValidation();
                } else {
                    result_dim = 4;
                    results = new double[8];
                    cvIndexFile = String.format("%s/%s_cold_start_4docs_fold_%d.txt", dataset, param.m_source, k);
                    analyzer.loadCVIndex(cvIndexFile);
                    //train
                    tModel.setTrainSet(analyzer.getDocsByCVIndex(3));//3 indicates training doc
                    System.out.format("[Info]train size = %d....\n", tModel.getTrainSize());

                    //test
                    for(int i = 0; i < result_dim; i++) {
                        if(i < result_dim-1)
                            tModel.setTestSet(analyzer.getDocsByCVIndex(i));
                        else {
                            tModel.setTestSet(analyzer.getDocsByCVIndex(0));
                            tModel.addTestSet(analyzer.getDocsByCVIndex(1));
                            tModel.addTestSet(analyzer.getDocsByCVIndex(2));
                        }
                        double[] cur_result = tModel.Evaluation2();
                        results[2*i] = cur_result[0];
                        results[2*i+1] = cur_result[1];
                        System.out.format("[Info]Part %d test size = %d...\n", i, tModel.getTestSize());
                    }
                }

                for(int i = 0; i < result_dim; i++){
                    perf[k][i] = results[2*i];
                    like[k][i] = results[2*i+1];
                }

                String resultFolder = outputFolder + k + "/";
                new File(resultFolder).mkdirs();
                tModel.printParameterAggregation(param.m_topk, resultFolder, param.m_topicmodel);
                tModel.printTopWords(param.m_topk);

                // label test by cvIndex==k
//                analyzer.maskDocByCVIndex(k);
//                tModel.setCorpus(analyzer.getCorpus());
//
//                System.out.format("====================\n[Info]Fold No. %d: \n", k);
//                double[] results = tModel.oneFoldValidation();
//                for(int i = 0; i < result_dim; i++){
//                    perf[k][i] = results[2*i];
//                    like[k][i] = results[2*i+1];
//                }
//
//                String resultFolder = outputFolder + k + "/";
//                new File(resultFolder).mkdirs();
//                tModel.printParameterAggregation(param.m_topk, resultFolder, param.m_topicmodel);
//                tModel.printTopWords(param.m_topk);
//
//                if(param.m_flag_tune){
//                    System.out.format("[Info]Tuning mode: only run one fold to save time.\n");
//                    break;
//                }
            }

            //output the performance statistics
            System.out.println();
            double mean = 0, var = 0;
            int[] invalid_label = new int[like.length];
            for(int j = 0; j < result_dim; j++) {
                System.out.format("Part %d -----------------", j);
                Arrays.fill(invalid_label, 0);
                for (int i = 0; i < like.length; i++) {
                    if (Double.isNaN(like[i][j]) || Double.isNaN(perf[i][j]) || perf[i][j] <= 0 )
                        invalid_label[i]=1;
                }
                int validLen = like.length - Utils.sumOfArray(invalid_label);
                System.out.format("Valid folds: %d\n", validLen);

                mean=0;
                var=0;
                for (int i = 0; i < like.length; i++) {
                    if (invalid_label[i]<1)
                        mean += like[i][j];
                }
                if(validLen>0)
                    mean /= validLen;
                for (int i = 0; i < like.length; i++) {
                    if (invalid_label[i]<1)
                        var += (like[i][j] - mean) * (like[i][j] - mean);
                }
                if(validLen>0)
                    var = Math.sqrt(var / validLen);
                System.out.format("[Stat]Loglikelihood %.3f+/-%.3f\n", mean, var);

                mean = 0;
                var = 0;
                for (int i = 0; i < perf.length; i++) {
                    if (invalid_label[i]<1)
                        mean += perf[i][j];
                }
                if(validLen>0)
                    mean /= validLen;
                for (int i = 0; i < perf.length; i++) {
                    if (invalid_label[i]<1)
                        var += (perf[i][j] - mean) * (perf[i][j] - mean);
                }
                if(validLen>0)
                    var = Math.sqrt(var / validLen);
                System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);
            }
        }
    }
}