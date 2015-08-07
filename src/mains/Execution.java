/**
 * 
 */
package mains;

import influence.PageRank;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;

import structures.Parameter;
import structures._Corpus;
import topicmodels.HTMM;
import topicmodels.LDA_Gibbs;
import topicmodels.LDA_Variational;
import topicmodels.LRHTMM;
import topicmodels.TopicModel;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import topicmodels.multithreads.LDA_Variational_multithread;
import topicmodels.multithreads.pLSA_multithread;
import Analyzer.Analyzer;
import Analyzer.DocAnalyzer;
import Analyzer.VctAnalyzer;
import Analyzer.jsonAnalyzer;
import Classifier.BaseClassifier;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFields;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.NaiveBayes;
import Classifier.supervised.PRLogisticRegression;
import Classifier.supervised.SVM;

/**
 * @author hongning
 *
 */
public class Execution  {
	static public void main(String[] args) throws IOException, ParseException {
		Parameter param = new Parameter(args);
		
		System.out.println(param.toString());
		
		String stnModel = (param.m_model.equals("HTMM")||param.m_model.equals("LRHTMM"))?param.m_stnModel:null;
		String posModel = (param.m_model.equals("HTMM")||param.m_model.equals("LRHTMM"))?param.m_posModel:null;
		
		_Corpus corpus;
		Analyzer analyzer;
		/***Load the data from vector file***/
		if (param.m_fvFile!=null && (new File(param.m_fvFile)).exists()) {
			analyzer = new VctAnalyzer(param.m_classNumber, param.m_lengthThreshold, param.m_featureFile);
			analyzer.LoadDoc(param.m_fvFile); //Load all the documents as the data set.
			
			corpus = analyzer.getCorpus();
		} else {
			/***Load the data from text file***/
			if (param.m_suffix.equals(".json"))
				analyzer = new jsonAnalyzer(param.m_tokenModel,param.m_classNumber, param.m_featureFile, param.m_Ngram, param.m_lengthThreshold, stnModel, posModel);	
			else
				analyzer = new DocAnalyzer(param.m_tokenModel, stnModel, posModel, param.m_classNumber,param.m_featureFile, param.m_Ngram, param.m_lengthThreshold);
			((DocAnalyzer)analyzer).setReleaseContent(!param.m_weightScheme.equals("PR"));
			if (param.m_featureFile==null) {
				/****Pre-process the data.*****/
				//Feture selection.
				System.out.println("Performing feature selection, wait...");
				param.m_featureFile = String.format("./data/Features/%s_fv.dat", param.m_featureSelection);
				param.m_featureStat = String.format("./data/Features/%s_fv_stat.dat", param.m_featureSelection);
				System.out.println(param.printFeatureSelectionConfiguration());
				
				((DocAnalyzer)analyzer).LoadStopwords(param.m_stopwords);
				analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
				analyzer.featureSelection(param.m_featureFile, param.m_featureSelection, param.m_startProb, param.m_endProb, param.m_DFthreshold); //Select the features.
			}
			
			//Collect vectors for documents.
			System.out.println("Creating feature vectors, wait...");			
			analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
			analyzer.setFeatureValues(param.m_featureValue, param.m_norm);
			analyzer.setTimeFeatures(param.m_window);
			
			corpus = analyzer.returnCorpus(param.m_featureStat);
		}
		
		if (param.m_weightScheme.equals("PR")) {
			System.out.println("Creating PageRank instance weighting, wait...");
			PageRank myPR = new PageRank(corpus, param.m_C, 100, 50, 1e-6);
			myPR.train(corpus.getCollection());
		}
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (param.m_style.equals("SUP")) {
			BaseClassifier model = null;
			
			if(param.m_model.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				model = new NaiveBayes(corpus);
			} else if(param.m_model.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				model = new LogisticRegression(corpus, param.m_C);
			} else if(param.m_model.equals("PR-LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start posterior regularized logistic regression, wait...");
				model = new PRLogisticRegression(corpus, param.m_C);
			} else if(param.m_model.equals("SVM")){
				//corpus.save2File("data/FVs/fvector.dat");
				System.out.println("Start SVM, wait...");
				model = new SVM(corpus, param.m_C);
			} else {
				System.out.println("Classifier has not been developed yet!");
				System.exit(-1);
			}
			
			model.setDebugOutput(param.m_debugOutput);
			model.crossValidation(param.m_CVFold, corpus);
		} else if (param.m_style.equals("SEMI")) {
			BaseClassifier model = null;
			
			if (param.m_model.equals("GF")) {
				System.out.println("Start Gaussian Field by matrix inversion, wait...");
				model = new GaussianFields(corpus, param.m_classifier, param.m_C,
					param.m_sampleRate, param.m_kUL, param.m_kUU);
			} else if (param.m_model.equals("GF-RW")) {
				System.out.println("Start Gaussian Field by random walk, wait...");
				model = new GaussianFieldsByRandomWalk(corpus, param.m_classifier, param.m_C,
					param.m_sampleRate, param.m_kUL, param.m_kUU, param.m_alpha, param.m_beta, param.m_converge, param.m_eta, param.m_weightedAvg);
			} else if (param.m_model.equals("GF-RW-ML")) {
				System.out.println("Start Gaussian Field with distance metric learning by random walk, wait...");
				model = new LinearSVMMetricLearning(corpus, param.m_classifier, param.m_C,
					param.m_sampleRate, param.m_kUL, param.m_kUU, param.m_alpha, param.m_beta, param.m_converge, param.m_eta, param.m_weightedAvg, 
					param.m_bound);
//				((LinearSVMMetricLearning)model).setMetricLearningMethod(false);
//				((LinearSVMMetricLearning)model).verification(param.m_CVFold, corpus, param.m_debugOutput);
			} else {
				System.out.println("Classifier has not been developed yet!");
				System.exit(-1);
			}
			
			model.setDebugOutput(param.m_debugOutput);
			model.crossValidation(param.m_CVFold, corpus);
		} else if (param.m_style.equals("TM")) {
			TopicModel model = null;
			if (param.m_model.equals("2topic")) {
				model = new twoTopic(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_lambda, analyzer.getBackgroundProb());
			} else if (param.m_model.equals("pLSA")) {
				if (param.m_multithread == false) {
					model = new pLSA(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
							param.m_lambda, param.m_numTopics, param.m_alpha);
				} else {
					model = new pLSA_multithread(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
							param.m_lambda, param.m_numTopics, param.m_alpha);
				}
				((pLSA)model).LoadPrior(param.m_priorFile, param.m_gamma);
			} else if (param.m_model.equals("vLDA")) {
				if (param.m_multithread == false) {
					model = new LDA_Variational(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
							param.m_lambda, param.m_numTopics, param.m_alpha, param.m_maxVarIterations, param.m_varConverge);
				} else {
					model = new LDA_Variational_multithread(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
							param.m_lambda, param.m_numTopics, param.m_alpha, param.m_maxVarIterations, param.m_varConverge);
				}
				
				((LDA_Variational)model).LoadPrior(param.m_priorFile, param.m_gamma);
			} else if (param.m_model.equals("gLDA")) {
					model = new LDA_Gibbs(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
							param.m_lambda, param.m_numTopics, param.m_alpha, param.m_burnIn, param.m_lag);
				
				((LDA_Gibbs)model).LoadPrior(param.m_priorFile, param.m_gamma);
			} else if (param.m_model.equals("HTMM")) {
				model = new HTMM(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_numTopics, param.m_alpha);
			} else if (param.m_model.equals("LRHTMM")) {				
				model = new LRHTMM(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_numTopics, param.m_alpha,
						param.m_C);
			} else {
				System.out.println("The specified topic model has not been developed yet!");
				System.exit(-1);
			}
			
			if (param.m_CVFold<=1) {
				model.EMonCorpus();
				model.printTopWords(10); // fixed: print top 10 words
			} else 
				model.crossValidation(param.m_CVFold);
		} else if (param.m_style.equals("FV")) {
			corpus.save2File(param.m_fvFile);
			System.out.format("Vectors saved to %s...\n", param.m_fvFile);
		} else 
			System.out.println("Learning paradigm has not developed yet!");
	}
}