package Classifier;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;

import libsvm.svm_model;

import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;

public abstract class BaseClassifier {
	protected _Corpus m_corpus;
	protected _SparseFeature[] m_sparseFeatures;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	protected int m_classNo; //The total number of classes.
	protected double[][] m_model; //The model contains total frequency for features /presences of features.
	protected double[][] m_sstat; //The probabilities of values in model.
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.
	protected int m_featureSize;
	
	//Used in NB and LR, will be overwritten.
	public void train(ArrayList<_Doc> trainSet){}
	//Used in SVM to train data, will be overwritten. Since java does not accept functions with same parameters 
	//and different return values, add a new variable.
	public svm_model train(ArrayList<_Doc> trainSet, boolean svmFlag){
		svm_model model = new svm_model();
		return model;
	}
	//Used in NB and LR, will be overwritten.
	public void test(ArrayList<_Doc> testSet){}	
	//Used in SVM, will be overwritten.
	public void test(ArrayList<_Doc> testSet, svm_model model){}
	

	//Constructor.
	public BaseClassifier() {
		this.m_corpus = new _Corpus();
		this.m_trainSet = new ArrayList<_Doc>();
		this.m_testSet = new ArrayList<_Doc>();
		this.m_classNo = 0;
		this.m_featureSize = 0;
	}

	// Constructor with parameters.
	public BaseClassifier(_Corpus c, int class_number, int featureSize) {
		this.m_corpus = c;
		this.m_trainSet = new ArrayList<_Doc>();
		this.m_testSet = new ArrayList<_Doc>();
		this.m_classNo = class_number;
		this.m_featureSize = featureSize;
		this.m_precisionsRecalls = new ArrayList<double[][]>();
	}
	
	//Add one more document array, which is a document folder to the train set.
	public void addTrainSet(ArrayList<_Doc> docs){
		for(_Doc doc: docs){
			this.m_trainSet.add(doc);
		}
	}
	
	//Set the test_set to be one of the folder.
	public void setTestSet(ArrayList<_Doc> docs){
		this.m_testSet = docs;
	}

	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c, int class_number){
		c.shuffle(k);
		int[] masks = c.getMasks();
		HashMap<Integer, ArrayList<_Doc>> k_folder = new HashMap<Integer, ArrayList<_Doc>>();

		// Set the hash map with documents.
		for (int i = 0; i < masks.length; i++) {
			_Doc doc = c.getCollection().get(i);
			if (k_folder.containsKey(masks[i])) {
				ArrayList<_Doc> temp = k_folder.get(masks[i]);
				temp.add(doc);
				k_folder.put(masks[i], temp);
			} else{
				ArrayList<_Doc> docs = new ArrayList<_Doc>();
				docs.add(doc);
				k_folder.put(masks[i], docs);
			}
		}

		// Set the train set and test set.
		for (int i = 0; i < k; i++) {
			this.setTestSet(k_folder.get(i));
			for (int j = 0; j < k; j++) {
				if (j != i) {
					this.addTrainSet(k_folder.get(j));
				}
			}
			// Train the data set to get the parameter.
			train(this.m_trainSet);
			test(this.m_testSet);
			this.m_trainSet.clear();
			//this.m_testSet.clear();
		}
		this.calculateMeanVariance(this.m_precisionsRecalls);	
	}
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(double[][] tpTable) {
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		for (int i = 0; i < this.m_classNo; i++) {
			PreRecOfOneFold[i][0] = tpTable[i][i] / sumOfColumn(tpTable, i);// Precision of the class.
			PreRecOfOneFold[i][1] = tpTable[i][i] / sumOfRow(tpTable, i);// Recall of the class.
		}
		return PreRecOfOneFold;
	}

	//Calculate the sum of a column in an array.
	public double sumOfColumn(double[][] tp, int i){
		double sum = 0;
		for(int j = 0; j < tp.length; j++){
			sum += tp[j][i];
		}
		return sum;
	}
	
	//Calculate the sum of a row in an array.
	public double sumOfRow(double[][] tp, int i){
		double sum = 0;
		for(int j = 0; j < tp[i].length; j++){
			sum += tp[i][j];
		}
		return sum;
	}
	
	//Calculate the mean and variance of precision and recall.
	public double[][] calculateMeanVariance(ArrayList<double[][]> prs){
		//Use the two-dimension array to represent the final result.
		double[][] metrix = new double[this.m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionMean = 0.0;
		double precisionVarSum = 0.0;
		double precisionVar = 0.0;

		double recallSum = 0.0;
		double recallMean = 0.0;
		double recallVarSum = 0.0;
		double recallVar = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < this.m_classNo; i++){
			precisionSum = 0;
			recallSum = 0;
			// Calculate the sum of precisions and recalls.
			for (int j = 0; j < prs.size(); j++) {
				precisionSum += prs.get(j)[i][0];
				recallSum += prs.get(i)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			precisionMean = precisionSum/prs.size();
			precisionMean =Double.parseDouble(new DecimalFormat("##.###").format(precisionMean));
			metrix[i][0] = precisionMean;
			recallMean = recallSum/prs.size();
			recallMean =Double.parseDouble(new DecimalFormat("##.###").format(recallMean));
			metrix[i][1] = recallMean;
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < this.m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < prs.size(); j++) {
				precisionVarSum += Math.pow((prs.get(j)[i][0] - metrix[i][0]), 2);
				recallVarSum += Math.pow((prs.get(j)[i][1] - metrix[i][1]), 2);
			}
			// Calculate the means of precisions and recalls.
			precisionVar = Math.sqrt(precisionVarSum/prs.size());
			precisionVar =Double.parseDouble(new DecimalFormat("##.###").format(precisionVar));
			metrix[i][2] = precisionVar;
			recallVar = Math.sqrt(recallVarSum/prs.size());
			recallVar =Double.parseDouble(new DecimalFormat("##.###").format(recallVar));
			metrix[i][3] = recallVar;
		}
		
		// The final output of the computation.
		System.out.println("*************************************************");
		System.out.println("The final result is as follows:");
		System.out.println("The total number of classes is " + this.m_classNo);
		
		for(int i = 0; i < this.m_classNo; i++){
			System.out.println("For class " + i + ":precision mean:" + metrix[i][0] + "\trecall mean:" + 
			metrix[i][1] + "\tprecision var:" + metrix[i][2] + "\trecall var:" + metrix[i][3]);
		}
		return metrix;
	}
	
}
