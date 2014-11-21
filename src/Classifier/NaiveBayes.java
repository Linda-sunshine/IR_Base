package Classifier;

import java.io.IOException;
import java.util.ArrayList;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Analyzer.DocAnalyzer;

public class NaiveBayes extends BaseClassifier {
	
	private double[] m_classProb;//p(c)
	private double[] m_classMember;//count(d|c)
	
	//Constructor.
	public NaiveBayes(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		this.m_model = new double [classNumber][featureSize];
		this.m_sstat = new double [classNumber][featureSize];
		this.m_classProb = new double [this.m_classNo];
		this.m_classMember = new double [this.m_classNo];
	}
	
	//Train the data set.
	public void train(ArrayList<_Doc> train_set){
		for(_Doc doc: train_set){
			int label = doc.getYLabel();
			this.m_classMember[label]++;
			for(_SparseFeature sf: doc.getSparse()){
				this.m_model[label][sf.getIndex()] += sf.getValue();
			}
		}
		calculateStat(this.m_model);
	}
	
	//Train the data set with term presence????
	//Is the numerator the total number of document in one class????
	//If is, I need to set different counters for different classes.
	public void trainPresence(ArrayList<_Doc> train_set){
		
		for(_Doc doc: train_set){
			int label = doc.getYLabel();
			for(_SparseFeature sf: doc.getSparse()){
				this.m_model[label][sf.getIndex()] += 1;
			}
		}
		calculateStat(this.m_model);
	}
	
	//Calculate the probabilities for different features in m_model;
	public void calculateStat(double[][] model){
		
		for(int i = 0; i < this.m_classNo; i++){
			this.m_classProb[i] = this.m_classMember[i]/this.m_trainSet.size();
		}
		for(int i = 0; i < model.length; i++){
			int sum = 0;
			for(int j = 0; j < model[i].length; j++){
				sum += model[i][j];
			}
			for(int j = 0; j < model[i].length; j++){
				this.m_sstat[i][j] = (this.m_model[i][j] + 1)/ (sum + this.m_featureSize);//add one smoothing
			}
		}
	}
	
	//Test the data set.
	public void test(ArrayList<_Doc> testSet){
		double[][] TPTable = new double [this.m_classNo][this.m_classNo];
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		double[] probs = new double[this.m_classNo];
		
		for(_Doc doc: testSet){
			for(int i = 0; i < this.m_classNo; i++){
				double probability = Math.log(this.m_classProb[i]);
				double[] sparseProbs = new double[doc.getSparse().length];
				double[] sparseValues = new double[doc.getSparse().length];
				
				//Construct probs array and values array first.
				for(int j = 0; j < doc.getSparse().length; j++){
					int index = doc.getSparse()[j].getIndex();
					sparseValues[j] = doc.getSparse()[j].getValue();
					sparseProbs[j] = m_sstat[i][index];
				}
				probability += Utils.sumLog(sparseProbs, sparseValues);
				probs[i] = probability;
			}
			doc.setPredictLabel(Utils.maxOfArrayIndex(probs)); //Set the predict label according to the probability of different classes.
			TPTable[doc.getPredictLabel()][doc.getYLabel()] +=1; //Compare the predicted label and original label, construct the TPTable.
		}
		PreRecOfOneFold = calculatePreRec(TPTable);
		this.m_precisionsRecalls.add(PreRecOfOneFold);
		this.m_classProb = new double [this.m_classNo];
		this.m_classMember = new double [this.m_classNo];
	}
	
/*****************************Main function*******************************/
	public static void main(String[] args) throws IOException{
		
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		_Corpus corpus = new _Corpus();
		int Ngram = 3; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be tfidf, BM25
		System.out.println(Ngram + " gram! " + featureValue + " is used to calculate feature value!");
		System.out.println("*******************************************************************");

		//The parameters used in loading files.
		String folder = "txt_sentoken";
		String suffix = ".txt";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/NB/NBFinal.txt"; //The destination of storing the final features with stats.
		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/NB/NBSelectedFeatures.txt";
//		String finalLocation = "/home/lin/Lin'sWorkSpace/IR_Base/NB/NBFinal.txt";
//		String featureLocation = "/home/lin/Lin'sWorkSpace/IR_Base/NB/NBSelectedFeatures.txt";

		String providedCV = "";
		//String featureSelection = "";
		//String providedCV = "Features.txt"; //Provided CV.
		String featureSelection = "MI"; //Feature selection method.
		double startProb = 0.455; // Used in feature selection, the starting point of the features.
		double endProb = 0.5; // Used in feature selection, the ending point of the feature.
		
		if( providedCV.isEmpty() && featureSelection.isEmpty()){	
			
			//Case 1: no provided CV, no feature selection.
			System.out.println("Case 1: no provided CV, no feature selection.");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, null, Ngram);
			System.out.println("Start loading files, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation);
			analyzer.setFeatureValues(corpus, analyzer, featureValue);
		} else if( !providedCV.isEmpty() && featureSelection.isEmpty()){
			
			//Case 2: provided CV, no feature selection.
			System.out.println("Case 2: provided CV, no feature selection.");
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, null, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation); 
			analyzer.setFeatureValues(corpus, analyzer, featureValue);
		} else if(providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 3: no provided CV, feature selection.
			System.out.println("Case 3: no provided CV, feature selection.");
			System.out.println("Start loading files to do feature selection, wait...");
			
//			If the feature selection is TS, we need to load the directory three times.
//			1. Load all the docs, get all the terms in the docs. Calculate the current document's similarity with other documents, find a max one??
//			2. Load again to do feature selection.
//			3. Load again to do classfication.
//			if(featureSelection.endsWith("TS")){
//				DocAnalyzer analyzer_1 = new DocAnalyzer(tokenModel, classNumber, null, null);
//				analyzer_1.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//				analyzer_1.calculateSimilarity();
//				//analyzer_1.featureSelection(featureLocation); //Select the features.
//			}
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection, Ngram);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);//featureLocation contains the selected features.
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, analyzer_2, featureValue);
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, featureSelection, Ngram);
			System.out.println("Case 4: provided CV, feature selection.");
			System.out.println("Start loading file to do feature selection, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation, startProb, endProb); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null, Ngram);
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			analyzer_2.setFeatureValues(corpus, analyzer_2, featureValue);
		}
		//Define a new naive bayes with the parameters.
		System.out.println("Start naive bayes, wait...");
		NaiveBayes myNB = new NaiveBayes(corpus, classNumber, featureSize);
		System.out.println("Start cross validaiton, wait...");
		myNB.crossValidation(10, corpus, classNumber);//Use the movie reviews for testing the codes.
	}

}
