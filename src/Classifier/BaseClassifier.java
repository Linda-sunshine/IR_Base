package Classifier;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;


public abstract class BaseClassifier {
	protected int m_classNo; //The total number of classes.
	protected int m_featureSize;
	protected _Corpus m_corpus;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	
	protected double[] m_cProbs;
	
	//for cross-validation
	protected double[][] m_TPTable;
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.

	public void train() {
		long start = System.currentTimeMillis();
		train(m_trainSet);
		System.out.format("%s training finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
	}
	
	public abstract void train(Collection<_Doc> trainSet);
	public abstract void test();
	public abstract int predict(_Doc doc);
	protected abstract void init(); // to be called before training starts
	
	// Constructor with parameters.
	public BaseClassifier(_Corpus c, int class_number, int featureSize) {
		m_classNo = class_number;
		m_featureSize = featureSize;
		m_corpus = c;
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new double [m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				if( masks[j]==i ) m_testSet.add(docs.get(j));
				else m_trainSet.add(docs.get(j));
			}
			train();
			test();
			m_trainSet.clear();
			m_testSet.clear();
		}
		calculateMeanVariance(m_precisionsRecalls);	
	}
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(double[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = tpTable[i][i] / (sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = tpTable[i][i] / (sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
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
		double[][] metrix = new double[m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionMean = 0.0;
		double precisionVarSum = 0.0;
		double precisionVar = 0.0;

		double recallSum = 0.0;
		double recallMean = 0.0;
		double recallVarSum = 0.0;
		double recallVar = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < m_classNo; i++){
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
		for (int i = 0; i < m_classNo; i++) {
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
		System.out.println("The total number of classes is " + m_classNo);
		
		for(int i = 0; i < m_classNo; i++){
			System.out.println("For class " + i + ":precision mean:" + metrix[i][0] + "\trecall mean:" + 
			metrix[i][1] + "\tprecision var:" + metrix[i][2] + "\trecall var:" + metrix[i][3]);
		}
		return metrix;
	}
}
