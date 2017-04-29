package mains;

import Analyzer.*;
import structures._Corpus;
import structures._Doc;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.LDA.LDA_Gibbs_test;
import topicmodels.correspondenceModels.*;

import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import topicmodels.twoTopic;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import topicmodels.correspondenceModels.*;

/**
 * Created by jetcai1900 on 4/29/17.
 */
public class CCTM_online_main {

    public static void main(String[] args) throws IOException, ParseException {

        int mb = 1024*1024;

        Runtime rTime = Runtime.getRuntime();
        System.out.println("totalMem\t:"+rTime.totalMemory()/mb);

        int classNumber = 9; //Define the number of classes in this Naive Bayes.
        int Ngram = 1; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        int minimunNumberofSentence = 2; // each document should have at least 2 sentences

        /*****parameters for the two-topic topic model*****/

        // CCTM_test, CCTM_Online_test,LDAGibbs4AC_test
        String topicmodel = "CCTM_test";

        String category = "tablet";
        int number_of_topics = 20;
        boolean loadNewEggInTrain = true; // false means in training there is no reviews from NewEgg
        boolean setRandomFold = true; // false means no shuffling and true means shuffling
        int loadAspectSentiPrior = 0; // 0 means nothing loaded as prior; 1 = load both senti and aspect; 2 means load only aspect

        double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = topicmodel.equals("LDA_Gibbs")?200:5.0;//these two parameters must be larger than 1!!!
        double converge = 1e-9, lambda = 0.9; // negative converge means do not need to check likelihood convergency
        int varIter = 10;
        double varConverge = 1e-5;
        int topK = 20, number_of_iteration = 200, crossV = 1;

        int gibbs_iteration = 500, gibbs_lag = 15;
        int displayLap = 10;

//		gibbs_iteration = 4;
//		gibbs_lag = 2;
//		displayLap = 2;

        double burnIn = 0.4;

        boolean sentence = false;

        // most popular items under each category from Amazon
        // needed for docSummary
        String tabletProductList[] = {"B008GFRDL0"};
        String cameraProductList[] = {"B005IHAIMA"};
        String phoneProductList[] = {"B00COYOAYW"};
        String tvProductList[] = {"B0074FGLUM"};

        /*****The parameters used in loading files.*****/
        String amazonFolder = "./data/amazon/tablet/topicmodel";
        String newEggFolder = "./data/NewEgg";
        String articleType = "Tech";
		articleType = "Medium";


        String articleFolder = String.format(
                "./data/ParentChildTopicModel/%sArticles",
                articleType);

        String commentFolder = String.format(
                "./data/ParentChildTopicModel/%sComments",
                articleType);

        String suffix = ".json";
        String tokenModel = "./data/Model/en-token.bin"; //Token model.

        String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_%s.txt", Ngram, articleType);
        String fvStatFile = String.format("./data/Features/fv_%dgram_stat_%s_%s.txt", Ngram, articleType, topicmodel);

        String aspectList = "./data/Model/aspect_"+ category + ".txt";
        String aspectSentiList = "./data/Model/aspect_sentiment_"+ category + ".txt";

        File rootFolder = new File("./data/results");
        if(!rootFolder.exists()){
            System.out.println("creating root directory"+rootFolder);
            rootFolder.mkdir();
        }

        SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyyMMdd-HHmm");
        String filePrefix = String.format("./data/results/%s", dateFormatter.format(new Date()));
        filePrefix = filePrefix + "-" + topicmodel + "-" + articleType;
        File resultFolder = new File(filePrefix);
        if (!resultFolder.exists()) {
            System.out.println("creating directory" + resultFolder);
            resultFolder.mkdir();
        }

        String outputFile = filePrefix + "/consoleOutput.txt";
        PrintStream printStream = new PrintStream(new FileOutputStream(
                outputFile));
        System.setOut(printStream);

        String infoFilePath = filePrefix + "/Information.txt";
        ////store top k words distribution over topic
        String topWordPath = filePrefix + "/topWords.txt";

        /*****Parameters in feature selection.*****/

        System.out.println("Performing feature selection, wait...");

        /***** parent child topic model *****/
        String rawFeatureFile = String.format(
                "./data/Features/rawFv_%dgram_topicmodel_%s.txt", Ngram,
                articleType);
//		topicmodelAnalyzer analyzer = new topicmodelAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, rawFeatureFile);

		ParentChildAnalyzer analyzer = new ParentChildAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold);
//		analyzer.LoadDirectory(commentFolder, suffix);
        if(topicmodel.equals("CCTM_Online_test")) {
            articleFolder = String.format(
                    "./data/ParentChildTopicModel/%sArticles_Online",
                    articleType);
            commentFolder = String.format(
                    "./data/ParentChildTopicModel/%sComments_Online",
                    articleType);
        }

        analyzer.LoadParentDirectory(articleFolder, suffix);
        analyzer.LoadChildDirectory(commentFolder, suffix);


        System.out.println("Creating feature vectors, wait...");

        analyzer.setFeatureValues(featureValue, norm);
        _Corpus c = analyzer.returnCorpus(fvStatFile); // Get the collection of all the documents.


        if (topicmodel.equals("2topic")) {
            twoTopic model = new twoTopic(number_of_iteration, converge, beta, c, lambda);

            if (crossV<=1) {
                for(_Doc d:c.getCollection()) {
                    model.inference(d);
                    model.printTopWords(topK);
                }
            } else
                model.crossValidation(crossV);
        } else {
            pLSA model = null;

            if (topicmodel.equals("pLSA")) {
                model = new pLSA_multithread(number_of_iteration, converge, beta, c,
                        lambda, number_of_topics, alpha);
            } else if (topicmodel.equals("LDA_Gibbs")) {
//				number_of_topics = 15;
                model = new LDA_Gibbs(gibbs_iteration, 0, beta, c, //in gibbs sampling, no need to compute log-likelihood during sampling
                        lambda, number_of_topics, alpha, burnIn, gibbs_lag);
            } else if (topicmodel.equals("LDA_Variational_multithread")) {
                model = new LDA_Variational_multithread(number_of_iteration, converge, beta, c,
                        lambda, number_of_topics, alpha, varIter, varConverge);
            } else if(topicmodel.equals("correspondence_LDA_Gibbs")){
                double ksi = 800;
                double tau = 0.7;
                model = new corrLDA_Gibbs(gibbs_iteration, 0, beta-1, c, //in gibbs sampling, no need to compute log-likelihood during sampling
                        lambda, number_of_topics, alpha-1, burnIn, gibbs_lag);
            }  else if (topicmodel.equals("LDAGibbs4AC_test")) {

                double ksi = 800;
                double tau = 0.7;
                model = new LDAGibbs4AC_test(gibbs_iteration, 0, beta-1, c,
                        lambda, number_of_topics, alpha-1, burnIn, gibbs_lag,
                        ksi, tau);

            } else if (topicmodel.equals("corrLDA_Gibbs_test")) {
                converge = 1e-3;
                double ksi = 800;
                double tau = 0.7;
                int newtonIter = 1000;
                double newtonConverge = 1e-3;

                model = new corrLDA_Gibbs_test(gibbs_iteration, 0,
                        beta-1, c, lambda, number_of_topics, alpha-1, burnIn,
                        gibbs_lag, ksi, tau);
            } else if (topicmodel.equals("DCMCorrLDA_test")) {
//				number_of_topics = 15;
                converge = 1e-3;
                int newtonIter = 50;
                double newtonConverge = 1e-3;
                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;
                model = new DCMCorrLDA_test(gibbs_iteration, converge,
                        beta - 1, c, lambda, number_of_topics, alpha - 1,
                        alphaC, burnIn, ksi, tau, gibbs_lag, newtonIter,
                        newtonConverge);
            } else if (topicmodel.equals("DCMCorrLDA_Multi_EM")) {
                converge = 1e-2;
                int newtonIter = 30;
                double newtonConverge = 1e-2;
                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;
                gibbs_iteration = 40;
                gibbs_lag = 20;
                model = new DCMCorrLDA_Multi_EM(gibbs_iteration, converge,
                        beta - 1, c, lambda, number_of_topics, alpha - 1,
                        alphaC, burnIn, gibbs_lag, ksi, tau, newtonIter,
                        newtonConverge);
            } else if(topicmodel.equals("weightedCorrespondenceModel_test")){
                beta = beta-1;
                alpha = alpha-1;
//				number_of_iteration = 2;
                double lbfgsConverge = varConverge;
                converge = 1e-6;
                model = new weightedCorrespondenceModel_test(number_of_iteration, converge, beta, c,
                        lambda, number_of_topics, alpha, varIter, varConverge, lbfgsConverge);
//
//				String priorFile = "./data/Features/" + articleType + "TopicWord.txt";
//				model.LoadPrior(priorFile, eta);eta
            }else if(topicmodel.equals("CorrDCMLDA_test")){
                converge = 1e-4;
                int newtonIter = 50;
                double newtonConverge = 1e-3;
                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;
                model = new CorrDCMLDA_test(gibbs_iteration, converge,
                        beta - 1, c, lambda, number_of_topics, alpha - 1,
                        alphaC, burnIn, ksi, tau, gibbs_lag, newtonIter,
                        newtonConverge);
            }else if(topicmodel.equals("CCTM_test")) {
                converge = 1e-3;
                int newtonIter = 100;
                double newtonConverge = 1e-3;

                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;
                model = new DCMCorrLDA_Multi_EM_test(gibbs_iteration, converge, beta - 1, c,
                        lambda, number_of_topics, alpha - 1, alphaC, burnIn, ksi, tau, gibbs_lag,
                        newtonIter, newtonConverge);
                String priorFile = "./data/Features/" + articleType + "TopicWord_"+number_of_topics+".txt";
                model.LoadPrior(priorFile, eta);
            }else if(topicmodel.equals("CorrDCMLDA_Multi_EM_test")){
                converge = 1e-3;
                int newtonIter = 100;
                double newtonConverge = 1e-3;

                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;
                model = new CorrDCMLDA_Multi_EM_test(gibbs_iteration, converge, beta - 1, c,
                        lambda, number_of_topics, alpha - 1, alphaC, burnIn, ksi, tau, gibbs_lag,
                        newtonIter, newtonConverge);
            }else if(topicmodel.equals("PriorCorrLDA_test")){
                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;

                model = new PriorCorrLDA_test(gibbs_iteration, 0,
                        beta-1, c, lambda, number_of_topics, alpha-1, alphaC, burnIn,
                        gibbs_lag, ksi, tau);
            }else if(topicmodel.equals("CCTM_Online_test")){
                converge = 1e-3;
                int newtonIter = 100;
                double newtonConverge = 1e-3;

                double ksi = 800;
                double tau = 0.7;
                double alphaC = 0.001;

                model = new CCTM_Online_test(gibbs_iteration, converge, beta - 1, c,
                        lambda, number_of_topics, alpha - 1, alphaC, burnIn, ksi, tau, gibbs_lag,
                        newtonIter, newtonConverge);
                String betaFile = "./data/Features/" + articleType + "_fullBetas.txt";
                String alphaFile = "./data/Features/" + articleType + "_alphas.txt";
                ((CCTM_Online_test)model).LoadWordDistributions(betaFile);
                ((CCTM_Online_test)model).LoadAlphas(alphaFile);
            }


            model.setDisplayLap(displayLap);
            model.setInforWriter(infoFilePath);

            if(loadAspectSentiPrior==1){
                System.out.println("Loading aspect-senti list from "+aspectSentiList);
                model.setSentiAspectPrior(true);
                model.LoadPrior(aspectSentiList, eta);
            } else if(loadAspectSentiPrior==2){
                System.out.println("Loading aspect list from "+aspectList);
                model.setSentiAspectPrior(false);
                model.LoadPrior(aspectList, eta);
            }else{
                System.out.println("No prior is added!!");
            }

            if (crossV<=1) {
                model.EMonCorpus();
                if(topWordPath == null)
                    model.printTopWords(topK);
                else
                    model.printTopWords(topK, topWordPath);
            } else {
                model.setRandomFold(setRandomFold);
                double trainProportion = 0.3;
                double testProportion = 1-trainProportion;
                model.setPerplexityProportion(testProportion);
                model.crossValidation(crossV);
                model.printTopWords(topK, topWordPath);
            }

            model.closeWriter();


        }
    }
}
