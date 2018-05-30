package mains;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import structures.TopicModelParameter;
import structures._Corpus;
import topicmodels.LDA.LDA_Gibbs;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA.pLSA_multithread;
import topicmodels.pLSA.pLSA;
import Analyzer.MultiThreadedReviewAnalyzer;

public class ETBIRExecution {

	public static void main(String[] args) throws IOException, ParseException {
		TopicModelParameter param = new TopicModelParameter(args);

		int classNumber = 6; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram.
		int lengthThreshold = 5; //Document length threshold
		int crossV = 5;
		boolean setRandomFold = true;
		int numberOfCores = Runtime.getRuntime().availableProcessors();

		String tokenModel = "./data/Model/en-token.bin";
		String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);
		String reviewFolder = String.format("%s/%s/byUser_70k_review/data/", param.m_prefix, param.m_source);

		System.out.println("[Info] Start preprocess textual data...");
		MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, numberOfCores, true, param.m_source);
		analyzer.loadUserDir(reviewFolder);
		_Corpus corpus = analyzer.getCorpus();

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
					param.m_lambda, param.m_number_of_topics, param.m_alpha, 10, 1e-5);
		} else if(param.m_topicmodel.equals("ETBIR")){
			tModel = new ETBIR(param.m_emIter, param.m_emConverge, param.m_beta, corpus, param.m_lambda,
					param.m_number_of_topics, param.m_alpha, param.m_varMaxIter, param.m_varConverge, param.m_sigma, param.m_rho);
		} else{
			System.out.println("The selected topic model has not developed yet!");
			return;
		}

		tModel.setDisplayLap(10);
		tModel.EMonCorpus();
		tModel.printTopWords(param.m_topk);

		// create result folder
		String resultDir = String.format("%s/%s_%d/", param.m_output, param.m_topicmodel, current);
		File resultFolder = new File(resultDir);
		if (!resultFolder.exists()) {
			System.out.println("[Info]Create directory " + resultFolder);
			resultFolder.mkdir();
		}
		((ETBIR) tModel).printParameterAggregation(param.m_topk, resultDir, param.m_topicmodel);
		if(crossV>1){
            tModel.setRandomFold(setRandomFold);
            double trainProportion = ((double)crossV - 1)/(double)crossV;
            double testProportion = 1-trainProportion;
            System.out.format("Begin %d-fold cross validation", crossV);
            tModel.setPerplexityProportion(testProportion);
            tModel.crossValidation(crossV);
        }
	}
}