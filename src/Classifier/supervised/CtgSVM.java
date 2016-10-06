package Classifier.supervised;

import java.util.ArrayList;
import structures._Doc;
import utils.Utils;

public class CtgSVM extends SVM {
	public CtgSVM(int classNo, int featureSize, double C) {
		super(classNo, featureSize, C);
		// TODO Auto-generated constructor stub
	}
	public void setTrainTestSets(ArrayList<_Doc> trainSet, ArrayList<_Doc> testSet){
		m_trainSet = trainSet;
		m_testSet = testSet;
		
	}
	public double test() {
		double acc = 0;
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			int pred = doc.getPredictLabel(), ans = doc.getYLabel();
			m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
			
			if (pred != ans) {
				if (m_debugOutput!=null && Math.random()<0.2)//try to reduce the output size
					debug(doc);
			} else {//also print out some correctly classified samples
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(doc);
				acc ++;
			}
		}
		calculatePreRec(m_TPTable);
		printConfusionMat();
		return 0;
	}

	public double[][] calculatePreRec(int[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][3];
		
		for (int i = 0; i < m_classNo; i++) {
			System.out.print("Class " + i + "\t");
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			System.out.print("pre: " + PreRecOfOneFold[i][0] + "\t");
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
			System.out.print("rec: " + PreRecOfOneFold[i][1] + "\t");
			PreRecOfOneFold[i][2] = 2*PreRecOfOneFold[i][0]*PreRecOfOneFold[i][1]/(PreRecOfOneFold[i][0]+PreRecOfOneFold[i][1]); // F1 of the class.
			System.out.print("F1: " + PreRecOfOneFold[i][2] + "\n");
		}
		
		for (int i = 0; i < m_classNo; i++) {			
			for(int j=0; j< m_classNo; j++) {
				m_confusionMat[i][j] += tpTable[i][j];
				tpTable[i][j] = 0; // clear the result in each fold
			}
		}
		return PreRecOfOneFold;
	}
}
