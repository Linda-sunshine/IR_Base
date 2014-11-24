package Classifier;

import java.util.ArrayList;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class LogisticRegression extends BaseClassifier{

	double[] m_beta;
	double m_lambda;
	
	public LogisticRegression(_Corpus c, int classNo, int featureSize){
		super(c, classNo, featureSize);
		this.m_beta = new double[classNo * (featureSize + 1)]; //Initialization.
		this.m_lambda = 0.5;//Initialize it to be 0.5.
	}
	
	public LogisticRegression(_Corpus c, int classNo, int featureSize, double lambda){
		super(c, classNo, featureSize);
		this.m_beta = new double[classNo * (featureSize + 1)]; //Initialization.
		this.m_lambda = lambda;//Initialize it to be 0.5.
	}

	/*
	 * Calculate the beta by using bfgs. In this method, we give a starting
	 * point and iterating the algorithm to find the minimum value for the beta.
	 * The input is the vector of feature[14], we need to pass the function
	 * value for the point, together with the gradient vector. When the iflag
	 * turns to 0, it finds the final point and we get the best beta.
	 */
	public void train(ArrayList<_Doc> docs){
		int[] iflag = {0};
		int[] iprint = { -1, 3 };
		double[] diag = new double[this.m_beta.length];

		try{
			do {
				LBFGS.lbfgs( this.m_classNo * (this.m_featureSize + 1), 5, this.m_beta, calculateFunction(this.m_beta, this.m_trainSet, this.m_lambda), calculateGradient(this.m_beta, this.m_trainSet, this.m_lambda), false, diag, iprint, 1e-3, 1e-7, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
	}
	
	//Calculate the value of the logit function.
	public double logitFunction(double[] beta, _SparseFeature[] sf){
		double sum = beta[0];
		double lfValue = 0;
		for(int i = 0; i < sf.length; i++){
			int index = sf[i].getIndex() + 1;
			sum += beta[index] * sf[i].getNormValue();
		}
		sum = Math.exp(-sum);
		lfValue = 1 / ( 1 + sum );
		return lfValue;
	}
	
	//This function is used to get a part of the array.
	public double[] trunc(double[] a, int index){
		double[] b = new double[this.m_featureSize + 1];
		System.arraycopy(a, index * (this.m_featureSize + 1), b, 0, this.m_featureSize + 1);		
		return b;
	}
	
	//This function is used to calculate Pij = P(Y=yi|X=xi) in multinominal LR.
	public double calculatelogPij(int Yi, double[] beta, _SparseFeature[] spXi){
		double logPij = 0;
		double[] xs = new double [this.m_classNo];
		double numerator = Utils.dotProduct(trunc(beta, Yi), spXi);
		double denominator = 0;
		for(int i = 0; i < this.m_classNo; i++){
			xs[i] = Utils.dotProduct(trunc(beta, i), spXi); 
		}
		denominator = Utils.logSumOfExponentials(xs);
		logPij = numerator - denominator;
		return logPij;
	}
	
	//This function is used to calculate the value with the new beta.
	public double calculateFunction(double[] beta, ArrayList<_Doc> trainSet, double lambda) {
		double totalSum = 0;
		//Calculate the sum of 
		for (_Doc doc: trainSet) {
			int Yi = doc.getYLabel();
			totalSum += calculatelogPij(Yi, beta, doc.getSparse());
			if (Double.isInfinite(totalSum)) {
				System.out.print("error!");
			}
		}
		// Add the L2 regularization.
		double L2 = 0;
		for (double b : beta)
			L2 += lambda * b * b;
		// LBFGS is used to calculate the minimum value while we are trying to calculate the maximum likelihood.
		return -totalSum + L2;
	}

	// This function is used to calculate the gradient descent of x, which is a
	// vector with the same dimension with x.
	public double[] calculateGradient(double[] beta, ArrayList<_Doc> docs, double lambda) {
		double[] gs = new double [beta.length]; // final gradient vector.
		double gValue = 0;
		double Pij = 0;
		double logPij = 0;
		//double Yi = 0;
		for(int i = 0; i < beta.length; i++){
			gs[i] += 2 * lambda * beta[i];
		}
		//The computation complexity is n*classNo.
		for (_Doc doc: docs) {
			for(int j = 0; j < this.m_classNo; j++){
				logPij = calculatelogPij(j, beta, doc.getSparse());//logP(Y=yi|X=xi)
				Pij = Math.exp(logPij);
				if (doc.getYLabel() == j){
					//gValue = 1.0 - Pij;
					gValue = Pij - 1.0;
				} else
					gValue = Pij;
				
				gs[j * (this.m_featureSize + 1)] += gValue;
				//(Yij - Pij) * Xi
				for(_SparseFeature sf: doc.getSparse()){
					int index = sf.getIndex();
					gs[j * (this.m_featureSize + 1) + index + 1] += gValue * sf.getNormValue();
				}
			}
		}
		return gs;
	}

	// Test the test set.
	public void test(ArrayList<_Doc> testSet) {
		double[][] TPTable = new double [this.m_classNo][this.m_classNo];
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		double[] probs = new double[this.m_classNo];
		
		for(_Doc doc: testSet){
			for(int i = 0; i < this.m_classNo; i++){
				probs[i] = calculatelogPij(i, this.m_beta, doc.getSparse());
				}
			doc.setPredictLabel(Utils.maxOfArrayIndex(probs)); //Set the predict label according to the probability of different classes.
			TPTable[doc.getPredictLabel()][doc.getYLabel()] +=1; //Compare the predicted label and original label, construct the TPTable.
		}
		
		PreRecOfOneFold = calculatePreRec(TPTable);
		this.m_precisionsRecalls.add(PreRecOfOneFold);
	}
}
