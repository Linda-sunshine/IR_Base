/**
 * 
 */
package Classifier;

import java.io.IOException;
import java.text.ParseException;

import structures.Parameter;
import structures._Corpus;
import Analyzer.Analyzer;
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
		
		Analyzer analyzer;
		if (param.m_suffix.equals(".json"))
			analyzer = new jsonAnalyzer(param.m_tokenModel, param.m_classNumber, param.m_featureFile, param.m_Ngram, param.m_lengthThreshold);
		else
			analyzer = new DocAnalyzer(param.m_tokenModel, param.m_classNumber, param.m_featureFile, param.m_Ngram, param.m_lengthThreshold);
		
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
		
		/********Choose different classification methods.*********/
		//Execute different classifiers.
		if (param.m_style.equals("SUP")) {
			if(param.m_classifier.equals("NB")){
				//Define a new naive bayes with the parameters.
				System.out.println("Start naive bayes, wait...");
				NaiveBayes myNB = new NaiveBayes(corpus, param.m_classNumber, featureSize + param.m_window);
				myNB.crossValidation(param.m_CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(param.m_classifier.equals("LR")){
				//Define a new logistics regression with the parameters.
				System.out.println("Start logistic regression, wait...");
				LogisticRegression myLR = new LogisticRegression(corpus, param.m_classNumber, featureSize + param.m_window, param.m_C);
				myLR.crossValidation(param.m_CVFold, corpus);//Use the movie reviews for testing the codes.
				
			} else if(param.m_classifier.equals("SVM")){
				//corpus.save2File("data/FVs/fvector.dat");
				System.out.println("Start SVM, wait...");
				SVM mySVM = new SVM(corpus, param.m_classNumber, featureSize + param.m_window, param.m_C);
				mySVM.crossValidation(param.m_CVFold, corpus);
				
			} else System.out.println("Classifier has not developed yet!");
		} else if (param.m_style.equals("TRANS")) {
			SemiSupervised mySemi = new SemiSupervised(corpus, param.m_classNumber, featureSize + param.m_window, param.m_classifier,
					param.m_sampleRate, param.m_kUL, param.m_kUU);
			mySemi.crossValidation(param.m_CVFold, corpus);
		} else System.out.println("Learning paradigm has not developed yet!");
	}
}
