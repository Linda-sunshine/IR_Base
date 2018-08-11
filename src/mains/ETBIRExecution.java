package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.util.HashSet;
import java.util.Set;

import Analyzer.BipartiteAnalyzer;
import structures.TopicModelParameter;
import structures._Corpus;
import structures._Doc;
import structures._Review;
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

public class ETBIRExecution {

	public static void main(String[] args) throws IOException, ParseException {
		TopicModelParameter param = new TopicModelParameter(args);

		int classNumber = 6; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram.
		int lengthThreshold = 5; //Document length threshold
		boolean setRandomFold = false;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		String tokenModel = "./data/Model/en-token.bin";
		String dataset = String.format("%s/%s/%s/", param.m_prefix, param.m_source, param.m_set);
		String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);
		String reviewFolder = dataset + "data/";
		String outputFolder = dataset + "output/" + param.m_crossV + "foldsCV" + "/";

		MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFile,
				Ngram, lengthThreshold, numberOfCores, true, param.m_source);
		if(setRandomFold==false)
			analyzer.setReleaseContent(false);//Remember to set it as false when generating crossfolders!!!
		analyzer.loadUserDir(reviewFolder);
		_Corpus corpus = analyzer.getCorpus();

		if(param.m_crossV>1 && setRandomFold==false){
			reviewFolder = dataset + param.m_crossV + "foldsCV/";
			//if no data, generate
            String cvFolder = reviewFolder + 0 + "/";
			File testFile = new File(cvFolder);
			if(!testFile.exists() && !testFile.isDirectory()){
				System.err.format("[Warning]Cross validation dataset %s not exist! Now generating...", cvFolder);
				BipartiteAnalyzer cv = new BipartiteAnalyzer(corpus); // split corpus into folds
				cv.analyzeCorpus();
                while(cv.splitCorpus(param.m_crossV,dataset + param.m_crossV + "foldsCV/")==false){
                    System.err.format("[Info]Split again...\n");
                }
			}
		}

		pLSA tModel = null;
		long current = System.currentTimeMillis();
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
        } else if(param.m_topicmodel.equals("ETBIR") || param.m_topicmodel.equals("ETBIR_User") || param.m_topicmodel.equals("ETBIR_Item")){
			tModel = new ETBIR_multithread(param.m_emIter, param.m_emConverge, param.m_beta, corpus, param.m_lambda,
					param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge, param.m_sigma, param.m_rho);
			if(param.m_topicmodel.equals("ETBIR_User"))
			    ((ETBIR_multithread) tModel).setMode("User");
			else if(param.m_topicmodel.equals("ETBIR_Item"))
			    ((ETBIR_multithread) tModel).setMode("Item");

			((ETBIR_multithread) tModel).setFlagGd(param.m_flag_gd);
            ((ETBIR_multithread) tModel).setFlagLambda(param.m_flag_fix_lambda);
		} else if(param.m_topicmodel.equals("CTM")){
			tModel = new CTM(param.m_emIter, param.m_emConverge, param.m_beta, corpus,
					param.m_lambda, param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge);
		} else{
			System.out.println("The selected topic model has not developed yet!");
			return;
		}

        tModel.setDisplayLap(1);
        new File(outputFolder).mkdirs();
        tModel.setInforWriter(outputFolder + param.m_topicmodel + "_info.txt");
        if (param.m_crossV<=1) {//just train
            tModel.EMonCorpus();
            tModel.printParameterAggregation(param.m_topk, outputFolder, param.m_topicmodel);
            tModel.closeWriter();
        } else if(setRandomFold == true){//cross validation with random folds
            tModel.setRandomFold(setRandomFold);
            double trainProportion = ((double)param.m_crossV - 1)/(double)param.m_crossV;
            double testProportion = 1-trainProportion;
            tModel.setPerplexityProportion(testProportion);
            tModel.crossValidation(param.m_crossV);
        } else{//cross validation with fixed folds
            double[] perf = new double[param.m_crossV];
            double[] like = new double[param.m_crossV];
            System.out.println("[Info]Start FIXED cross validation...");
            for(int k = 0; k <param.m_crossV; k++){
                analyzer.getCorpus().reset();
                //load test set
                String testFolder = reviewFolder + k + "/";
                analyzer.loadUserDir(testFolder);
                for(_Doc d : analyzer.getCorpus().getCollection()){
                    d.setType(_Review.rType.TEST);
                }
                //load train set
                for(int i = 0; i < param.m_crossV; i++){
                    if(i!=k){
                        String trainFolder = reviewFolder + i + "/";
                        analyzer.loadUserDir(trainFolder);
                    }
                }
                tModel.setCorpus(analyzer.getCorpus());

                System.out.format("====================\n[Info]Fold No. %d: ", k);
                double[] results = tModel.oneFoldValidation();
                perf[k] = results[0];
                like[k] = results[1];

                String resultFolder = outputFolder + k + "/";
                new File(resultFolder).mkdirs();
                tModel.printParameterAggregation(param.m_topk, resultFolder, param.m_topicmodel);
                tModel.printTopWords(param.m_topk);
            }

            //output the performance statistics
            Set invalid = new HashSet();
            for(int i = 0; i < like.length; i++){
                if(Double.isNaN(like[i]) || Double.isNaN(perf[i]))
                    invalid.add(i);
            }
            int validLen = like.length - invalid.size();
            System.out.format("[Info]Valid folds: %d\n", validLen);

            double mean = 0, var = 0;
            for(int i = 0; i < like.length; i++){
                if(!invalid.contains(i))
                    mean += like[i];
            }
            mean /= validLen;
            for(int i=0; i<like.length; i++) {
                if(!invalid.contains(i))
                    var += (like[i] - mean) * (like[i] - mean);
            }
            var = Math.sqrt(var/validLen);
            System.out.format("[Stat]Loglikelihood %.3f+/-%.3f\n", mean, var);

            mean = 0;
            var = 0;
            for(int i = 0; i < perf.length; i++){
                if(!invalid.contains(i))
                    mean += perf[i];
            }
            mean /= validLen;
            for(int i=0; i<perf.length; i++) {
                if(!invalid.contains(i))
                    var += (perf[i] - mean) * (perf[i] - mean);
            }
            var = Math.sqrt(var/validLen);
            System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);
        }
    }
}