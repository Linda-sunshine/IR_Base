/**
 * 
 */
package Classifier;

import influence.PageRank;

import java.io.IOException;
import java.text.ParseException;

import structures.Parameter;
import structures._Corpus;
import topicmodels.HTMM;
import topicmodels.LRHTMM;
import topicmodels.TopicModel;
import topicmodels.pLSA;
import topicmodels.twoTopic;
import Analyzer.DocAnalyzer;
import Analyzer.jsonAnalyzer;

/**
 * @author hongning
 *
 */
public class Execution  {
	static public void main(String[] args) throws IOException, ParseException {
		Parameter param = new Parameter(args);
		
		System.out.println(param.toString());
		
		String stnModel = (param.m_model.equals("HTMM")||param.m_model.equals("LRHTMM"))?param.m_stnModel:null;
		
		DocAnalyzer analyzer;
		if (param.m_suffix.equals(".json"))
			analyzer = new jsonAnalyzer(param.m_tokenModel, param.m_classNumber, param.m_featureFile, param.m_Ngram, param.m_lengthThreshold, stnModel);
		else
			analyzer = new DocAnalyzer(param.m_tokenModel, stnModel, param.m_classNumber, param.m_featureFile, param.m_Ngram, param.m_lengthThreshold);
		analyzer.setReleaseContent(!param.m_weightScheme.equals("PR"));
		
		/****Pre-process the data.*****/
		if (param.m_featureFile==null) {
		//Feture selection.
			System.out.println("Performing feature selection, wait...");
			param.m_featureFile = String.format("./data/Features/%s_fv.dat", param.m_featureSelection);
			param.m_featureStat = String.format("./data/Features/%s_fv_stat.dat", param.m_featureSelection);
			System.out.println(param.printFeatureSelectionConfiguration());
			
			((DocAnalyzer)analyzer).LoadStopwords(param.m_stopwords);
			analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
			analyzer.featureSelection(param.m_featureFile, param.m_featureSelection, param.m_startProb, param.m_endProb, param.m_DFthreshold); //Select the features.
			analyzer.reset();//clear memory for future feature construction
		}
		
		//Collect vectors for documents.
		System.out.println("Creating feature vectors, wait...");
		
		analyzer.LoadDirectory(param.m_folder, param.m_suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(param.m_featureValue, param.m_norm);
		analyzer.setTimeFeatures(param.m_window);
		
		int featureSize = analyzer.getFeatureSize();
		_Corpus corpus = analyzer.returnCorpus(param.m_featureStat);
		
		if (param.m_weightScheme.equals("PR")) {
			System.out.println("Creating PageRank instance weighting, wait...");
			PageRank myPR = new PageRank(corpus, param.m_classNumber, featureSize + param.m_window, param.m_C, 100, 50, 1e-6);
			myPR.train(corpus.getCollection());
		}
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (param.m_style.equals("SUP")) {
			BaseClassifier model = null;
			
			if(param.m_model.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				model = new NaiveBayes(corpus, param.m_classNumber, featureSize + param.m_window);
			} else if(param.m_model.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				model = new LogisticRegression(corpus, param.m_classNumber, featureSize + param.m_window, param.m_C);
			} else if(param.m_model.equals("SVM")){
				//corpus.save2File("data/FVs/fvector.dat");
				System.out.println("Start SVM, wait...");
				model = new SVM(corpus, param.m_classNumber, featureSize + param.m_window, param.m_C);
			} else {
				System.out.println("Classifier has not developed yet!");
				System.exit(-1);
			}
			model.crossValidation(param.m_CVFold, corpus);
		} else if (param.m_style.equals("TRANS")) {
			SemiSupervised mySemi = new SemiSupervised(corpus, param.m_classNumber, featureSize + param.m_window, param.m_model,
					param.m_sampleRate, param.m_kUL, param.m_kUU);
			mySemi.crossValidation(param.m_CVFold, corpus);
		} else if (param.m_style.equals("TM")) {
			TopicModel model = null;
			
			if (param.m_model.equals("2topic")) {
				model = new twoTopic(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_lambda, analyzer.getBackgroundProb());
			} else if (param.m_model.equals("pLSA")) {			
				model = new pLSA(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_lambda, analyzer.getBackgroundProb(), 
						param.m_numTopics, param.m_alpha);
			} else if (param.m_model.equals("HTMM")) {
				model = new HTMM(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_numTopics, param.m_alpha);
			} else if (param.m_model.equals("LRHTMM")) {
				corpus.setStnFeatures();
				
				model = new LRHTMM(param.m_maxmIterations, param.m_converge, param.m_beta, corpus, 
						param.m_numTopics, param.m_alpha,
						param.m_lambda);
			} else {
				System.out.println("The specified topic model has not developed yet!");
				System.exit(-1);
			}
			
			model.crossValidation(param.m_CVFold);
		} else 
			System.out.println("Learning paradigm has not developed yet!");
	}
}
