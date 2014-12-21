package Classifier;
import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import utils.Utils;


public abstract class BaseClassifier {
	protected int m_classNo; //The total number of classes.
	protected int m_featureSize;
	protected _Corpus m_corpus;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	
	protected double[] m_cProbs;
	
	//for cross-validation
	protected int[][] m_confusionMat, m_TPTable;//confusion matrix over all folds, prediction table in each fold
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.

	public void train() {
		train(m_trainSet);
	}
	
	public abstract void train(Collection<_Doc> trainSet);
	public abstract int predict(_Doc doc);
	protected abstract void init(); // to be called before training starts
	
	public void test() {
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			m_TPTable[doc.getPredictLabel()][doc.getYLabel()] += 1; //Compare the predicted label and original label, construct the TPTable.
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
	}
	
	// Constructor with parameters.
	public BaseClassifier(_Corpus c, int class_number, int featureSize) {
		m_classNo = class_number;
		m_featureSize = featureSize;
		m_corpus = c;
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
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
			
			long start = System.currentTimeMillis();
			train();
			test();
			System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
			m_trainSet.clear();
			m_testSet.clear();
		}
		calculateMeanVariance(m_precisionsRecalls);	
	}
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(int[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
			
			for(int j=0; j< m_classNo; j++) {
				m_confusionMat[i][j] += tpTable[i][j];
				tpTable[i][j] = 0; // clear the result in each fold
			}
		}
		return PreRecOfOneFold;
	}
	
	public void printConfusionMat() {
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%d", i);
		
		double total = 0, correct = 0;
		double[] columnSum = new double[m_classNo], prec = new double[m_classNo];
		System.out.println("\tP");
		for(int i=0; i<m_classNo; i++){
			System.out.format("%d", i);
			double sum = 0; // row sum
			for(int j=0; j<m_classNo; j++) {
				System.out.format("\t%d", m_confusionMat[i][j]);
				sum += m_confusionMat[i][j];
				columnSum[j] += m_confusionMat[i][j];
				total += m_confusionMat[i][j];
			}
			correct += m_confusionMat[i][i];
			prec[i] = m_confusionMat[i][i]/sum;
			System.out.format("\t%.4f\n", prec[i]);
		}
		
		System.out.print("R");
		for(int i=0; i<m_classNo; i++){
			columnSum[i] = m_confusionMat[i][i]/columnSum[i]; // recall
			System.out.format("\t%.4f", columnSum[i]);
		}
		System.out.format("\t%.4f", correct/total);
		
		System.out.print("\nF1");
		for(int i=0; i<m_classNo; i++)
			System.out.format("\t%.4f", 2.0 * columnSum[i] * prec[i] / (columnSum[i] + prec[i]));
		System.out.println();
	}
	
	//Calculate the mean and variance of precision and recall.
	public double[][] calculateMeanVariance(ArrayList<double[][]> prs){
		//Use the two-dimension array to represent the final result.
		double[][] metrix = new double[m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionVarSum = 0.0;
		double recallSum = 0.0;
		double recallVarSum = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < m_classNo; i++){
			precisionSum = 0;
			recallSum = 0;
			// Calculate the sum of precisions and recalls.
			for (int j = 0; j < prs.size(); j++) {
				precisionSum += prs.get(j)[i][0];
				recallSum += prs.get(j)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][0] = precisionSum/prs.size();
			metrix[i][1] = recallSum/prs.size();
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < prs.size(); j++) {
				precisionVarSum += (prs.get(j)[i][0] - metrix[i][0])*(prs.get(j)[i][0] - metrix[i][0]);
				recallVarSum += (prs.get(j)[i][1] - metrix[i][1])*(prs.get(j)[i][1] - metrix[i][1]);
			}
			// Calculate the means of precisions and recalls.
			metrix[i][2] = Math.sqrt(precisionVarSum/prs.size());
			metrix[i][3] = Math.sqrt(recallVarSum/prs.size());
		}
		
		// The final output of the computation.
		System.out.println("*************************************************");
		System.out.println("The final result is as follows:");
		System.out.println("The total number of classes is " + m_classNo);
		
		for(int i = 0; i < m_classNo; i++)
			System.out.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][1], metrix[i][2], metrix[i][3]);
		
		printConfusionMat();
		return metrix;
	}
}
