package sto2;

import java.util.List;
import java.util.TreeSet;

import util.DoubleMatrix;
import util.IntegerMatrix;

public class STO2Util {

	public static DoubleMatrix [] calculatePhi(IntegerMatrix [] matrixSWT, int [][] sumSTW, double [] betas, double [] sumBeta, List<TreeSet<Integer>> sentiWordsList) {  // Beta
		System.out.println("Calculating Phi...");
		int numSenti = matrixSWT.length;
		int numWords = matrixSWT[0].getNumOfRow();
		int numTopics = matrixSWT[0].getNumOfColumn();
		
		DoubleMatrix [] Phi = new DoubleMatrix[numSenti];
		
		Integer [] wordLexicons = new Integer[numWords];
		for (int w = 0; w < numWords; w++) {
			wordLexicons[w] = null;
			for (int s = 0; s < numSenti; s++) {
				if (sentiWordsList.get(s).contains(w)) wordLexicons[w] = s;
			}
		}
		for (int s = 0; s < numSenti; s++) {
			Phi[s] = new DoubleMatrix(numWords, numTopics);
			for (int w = 0; w < numWords; w++) {
				for (int t = 0; t < numTopics; t++) {
					double beta;
					if (wordLexicons[w] == null) beta = betas[0];
					else if (wordLexicons[w] == s) beta = betas[1];
					else beta = betas[2];
					
					double value = (matrixSWT[s].getValue(w, t) + beta) / (sumSTW[s][t] + sumBeta[s]);
					Phi[s].setValue(w, t, value);
				}
			}
		}
		
		return Phi;
	}
	
	public static DoubleMatrix [] calculateTheta(IntegerMatrix [] matrixSDT, int [][] sumDST, double alpha, double sumAlpha) {
		System.out.println("Calculating Theta...");
		int numSenti = matrixSDT.length;
		int numDocs = matrixSDT[0].getNumOfRow();
		int numTopics = matrixSDT[0].getNumOfColumn();
		
		DoubleMatrix [] Theta = new DoubleMatrix[numSenti];
	
		for (int s = 0; s < numSenti; s++) {
			Theta[s] = new DoubleMatrix(numDocs, numTopics);
			for (int d = 0; d < numDocs; d++) {
				for (int t = 0; t < numTopics; t++) {
					double value = (matrixSDT[s].getValue(d,t) + alpha) / (sumDST[d][s] + sumAlpha);
					Theta[s].setValue(d, t, value);
				}
			}
		}
		return Theta;
	}
	
	public static DoubleMatrix calculatePi(IntegerMatrix matrixDS, int [] sumDS, double [] gammas, double sumGamma) {
		System.out.println("Calculating Pi...");
		int numDocs = matrixDS.getNumOfRow();
		int numSenti = matrixDS.getNumOfColumn();
		
		DoubleMatrix Pi = new DoubleMatrix(numDocs, numSenti);
		
		for (int d = 0; d < numDocs; d++) {
			for (int s = 0; s < numSenti; s++) {
				double value = (matrixDS.getValue(d,s) + gammas[s]) / (sumDS[d] + sumGamma);
				Pi.setValue(d, s, value);
			}
		}
		return Pi;
	}
}
