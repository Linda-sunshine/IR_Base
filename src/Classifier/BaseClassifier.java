package Classifier;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;

import Classifier.supervised.liblinear.Feature;

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

	protected String m_debugOutput; // set up debug output (default: no debug output)
	protected BufferedWriter m_debugWriter; // debug output writer
	
	protected double[][][] m_purityStat;
	
	protected Feature[][] m_fvs; //The data instances for testing svm, we don't need it in real L2R.
	protected Integer[] m_ys; //The data labels for testing svm.
	protected int m_kFold; //k-fold cross validation.
	public void train() {
		train(m_trainSet);
	}
	
	public abstract void train(Collection<_Doc> trainSet);
	public abstract int predict(_Doc doc);//predict the class label
	public abstract double score(_Doc d, int label);//output the prediction score
	protected abstract void init(); // to be called before training starts
	protected abstract void debug(_Doc d);

	public double test() {
		double acc = 0;
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			int pred = doc.getPredictLabel(), ans = doc.getYLabel();
			m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
			
			if (pred != ans) {
				if (m_debugOutput!=null)
					debug(doc);
			} else {//also print out some correctly classified samples
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(doc);
				acc ++;
			}
		}
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		return acc /m_testSet.size();
	}
	
	public String getF1String() {
		double[][] PRarray = m_precisionsRecalls.get(m_precisionsRecalls.size()-1);
		StringBuffer buffer = new StringBuffer(128);
		for(int i=0; i<PRarray.length; i++) {
			double p = PRarray[i][0], r = PRarray[i][1];
			buffer.append(String.format("%d:%.3f ", i, 2*p*r/(p+r)));
		}
		return buffer.toString().trim();
	}
	
	// Constructor with given corpus.
	public BaseClassifier(_Corpus c) {
		m_classNo = c.getClassSize();
		m_featureSize = c.getFeatureSize();
		m_corpus = c;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_debugOutput = null;
	}
	
	// Constructor with given dimensions
	public BaseClassifier(int classNo, int featureSize) {
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_corpus = null;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_debugOutput = null;
	}
	
	public void setDebugOutput(String filename) {
		if (filename==null || filename.isEmpty())
			return;
		
		File f = new File(filename);
		if(!f.isDirectory()) { 
			if (f.exists()) 
				f.delete();
			m_debugOutput = filename;
		} else {
			System.err.println("Please specify a correct path for debug output!");
		}	
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		m_kFold = k;
		m_purityStat = new double[m_kFold][4][3];
		try {
			if (m_debugOutput!=null){
				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
				m_debugWriter.write(this.toString() + "\n");
			}
			c.shuffle(m_kFold);
//			c.mastInOrder(k);
			int[] masks = c.getMasks();
			ArrayList<_Doc> docs = c.getCollection();
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < m_kFold; i++) {
				for (int j = 0; j < masks.length; j++) {
					//two fold for training, eight fold for testing.
					if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k 
						m_trainSet.add(docs.get(j));
					else
						m_testSet.add(docs.get(j));
					
					//One fold for training, nine fold for testing.
//					if( masks[j]==i ) // || masks[j]==(i+3)%k 
//						m_trainSet.add(docs.get(j));
//					else
//						m_testSet.add(docs.get(j));
				}
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();				
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
				m_trainSet.clear();
				m_testSet.clear();
			}
			calculateMeanVariance();	
			calculateMeanPurity();
			if (m_debugOutput!=null)
				m_debugWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
//	public void crossValidation(int k, _Corpus c){
//		m_purityStat = new double[k][4][3];
//		try {
//			if (m_debugOutput!=null){
//				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
//				m_debugWriter.write(this.toString() + "\n");
//			}
////			c.shuffle(k);
//			c.mastInOrder(k);
//			int[] masks = c.getMasks();
//			ArrayList<_Doc> docs = c.getCollection();
//			//Use this loop to iterate all the ten folders, set the train set and test set.
//			for (int i = 0; i < k; i++) {
//				for (int j = 0; j < masks.length; j++) {
//					//more for testing
//					if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k 
//						m_trainSet.add(docs.get(j));
//					else
//						m_testSet.add(docs.get(j));
//				}
//				
////				long start = System.currentTimeMillis();
//				train();
//				save2File("./data/RankSVMDataFile1027.csv", 1000);
////				double accuracy = test();				
////				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
////				m_trainSet.clear();
////				m_testSet.clear();
//			}
////			calculateMeanVariance();	
////			calculateMeanPurity();
////			if (m_debugOutput!=null)
////				m_debugWriter.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}
	
	//Print part of the instances for sanity check.
	public void save2File(String filename, int n) throws FileNotFoundException{
		PrintWriter printer = new PrintWriter(new File(filename));
		for(int i=0; i<n; i++){
			printer.print(m_ys[i]+",");//print the label first.
			Feature[] tmp = m_fvs[i];
			for(int j=0; j<tmp.length-1; j++){
				printer.print(tmp[j].getIndex()+","+tmp[j].getValue()+",");
			}
			printer.print(tmp[tmp.length-1].getIndex()+","+tmp[tmp.length-1].getValue()+"\n");
		}
		printer.close();
	}
	
	//Calculate the mean of purity. added by Lin.
	public void calculateMeanPurity(){
		System.out.println("\nQuery\tDocs\tP@5\tP@10\tP@20");
		int folder = m_purityStat.length;
		double[][] puritySum = new double[4][3];
		for(int i=0; i<4; i++){
			for(int j=0; j<3; j++){
				puritySum[i][j] = sumPurity(i, j);
			}
		}
		System.out.format("Pos\tU\t%.3f\t%.3f\t%.3f\n", puritySum[0][0]/folder, puritySum[0][1]/folder, puritySum[0][2]/folder);
		System.out.format("Pos\tL\t%.3f\t%.3f\t%.3f\n", puritySum[1][0]/folder, puritySum[1][1]/folder, puritySum[1][2]/folder);
		System.out.format("Neg\tU\t%.3f\t%.3f\t%.3f\n", puritySum[2][0]/folder, puritySum[2][1]/folder, puritySum[2][2]/folder);
		System.out.format("Neg\tL\t%.3f\t%.3f\t%.3f\n\n", puritySum[3][0]/folder, puritySum[3][1]/folder, puritySum[3][2]/folder);
	}
	
	public double sumPurity(int i, int j){
		double sum = 0;
		for(int k=0; k<m_purityStat.length; k++){
			sum += m_purityStat[k][i][j];
		}
		return sum;
	}
	abstract public void saveModel(String modelLocation);
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(int[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
		}
		
		for (int i = 0; i < m_classNo; i++) {			
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
	public double[][] calculateMeanVariance(){
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
			for (int j = 0; j < m_precisionsRecalls.size(); j++) {
				precisionSum += m_precisionsRecalls.get(j)[i][0];
				recallSum += m_precisionsRecalls.get(j)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][0] = precisionSum/m_precisionsRecalls.size();
			metrix[i][1] = recallSum/m_precisionsRecalls.size();
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < m_precisionsRecalls.size(); j++) {
				precisionVarSum += (m_precisionsRecalls.get(j)[i][0] - metrix[i][0])*(m_precisionsRecalls.get(j)[i][0] - metrix[i][0]);
				recallVarSum += (m_precisionsRecalls.get(j)[i][1] - metrix[i][1])*(m_precisionsRecalls.get(j)[i][1] - metrix[i][1]);
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][2] = Math.sqrt(precisionVarSum/m_precisionsRecalls.size());
			metrix[i][3] = Math.sqrt(recallVarSum/m_precisionsRecalls.size());
		}
		
		// The final output of the computation.
		System.out.println("*************************************************");
		System.out.format("The final result of %s is as follows:\n", this.toString());
		System.out.println("The total number of classes is " + m_classNo);
		
		for(int i = 0; i < m_classNo; i++)
			System.out.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][2], metrix[i][1], metrix[i][3]);
		
		printConfusionMat();
		return metrix;
	}
}
