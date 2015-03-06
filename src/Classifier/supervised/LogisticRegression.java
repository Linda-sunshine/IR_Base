package Classifier.supervised;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class LogisticRegression extends BaseClassifier {

	double[] m_beta;
	double[] m_g, m_diag;
	double[] m_cache;
	double m_lambda;
	
	public LogisticRegression(_Corpus c, int classNo, int featureSize){
		super(c, classNo, featureSize);
		m_beta = new double[classNo * (featureSize + 1)]; //Initialization.
		m_g = new double[m_beta.length];
		m_diag = new double[m_beta.length];
		m_cache = new double[classNo];
		m_lambda = 0.5;//Initialize it to be 0.5.
	}
	
	public LogisticRegression(_Corpus c, int classNo, int featureSize, double lambda){
		super(c, classNo, featureSize);
		m_beta = new double[classNo * (featureSize + 1)]; //Initialization.
		m_g = new double[m_beta.length];
		m_diag = new double[m_beta.length];
		m_cache = new double[classNo];
		m_lambda = lambda;
	}
	
	@Override
	public String toString() {
		return String.format("Logistic Regression[C:%d, F:%d, L:%.2f]", m_classNo, m_featureSize, m_lambda);
	}
	
	@Override
	protected void init() {
		Arrays.fill(m_beta, 0);
		Arrays.fill(m_diag, 0);
	}

	/*
	 * Calculate the beta by using bfgs. In this method, we give a starting
	 * point and iterating the algorithm to find the minimum value for the beta.
	 * The input is the vector of feature[14], we need to pass the function
	 * value for the point, together with the gradient vector. When the iflag
	 * turns to 0, it finds the final point and we get the best beta.
	 */	
	@Override
	public void train(Collection<_Doc> trainSet) {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue;
		int fSize = m_beta.length;
		
		init();
		try{
			do {
				fValue = calcFuncGradient(trainSet);
				LBFGS.lbfgs(fSize, 6, m_beta, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
	}
	
	//This function is used to calculate Pij = P(Y=yi|X=xi) in multi-class LR.
	protected void calcPosterior(_SparseFeature[] spXi, double[] prob){
		int offset = 0;
		for(int i = 0; i < m_classNo; i++){			
			offset = i * (m_featureSize + 1);
			prob[i] = Utils.dotProduct(m_beta, spXi, offset);			
		}
		
		double logSum = Utils.logSumOfExponentials(prob);
		for(int i = 0; i < m_classNo; i++)
			prob[i] = Math.exp(prob[i] - logSum);
	}
	
	//This function is used to calculate the value and gradient with the new beta.
	protected double calcFuncGradient(Collection<_Doc> trainSet) {		
		double gValue = 0, fValue = 0;
		double Pij = 0, logPij = 0;

		// Add the L2 regularization.
		double L2 = 0, b;
		for(int i = 0; i < m_beta.length; i++) {
			b = m_beta[i];
			m_g[i] = 2 * m_lambda * b;
			L2 += b * b;
		}
		
		//The computation complexity is n*classNo.
		int Yi;
		_SparseFeature[] fv;
		double weight;
		for (_Doc doc: trainSet) {
			Yi = doc.getYLabel();
			fv = doc.getSparse();
			weight = doc.getWeight();
			
			//compute P(Y=j|X=xi)
			calcPosterior(fv, m_cache);
			for(int j = 0; j < m_classNo; j++){
				Pij = m_cache[j];
				logPij = Math.log(Pij);
				if (Yi == j){
					gValue = Pij - 1.0;
					fValue += logPij * weight;
				} else
					gValue = Pij;
				gValue *= weight;//weight might be different for different documents
				
				int offset = j * (m_featureSize + 1);
				m_g[offset] += gValue;
				//(Yij - Pij) * Xi
				for(_SparseFeature sf: fv)
					m_g[offset + sf.getIndex() + 1] += gValue * sf.getValue();
			}
		}
			
		// LBFGS is used to calculate the minimum value while we are trying to calculate the maximum likelihood.
		return m_lambda*L2 - fValue;
	}
	
	@Override
	public int predict(_Doc doc) {
		_SparseFeature[] fv = doc.getSparse();
		for(int i = 0; i < m_classNo; i++)
			m_cache[i] = Utils.dotProduct(m_beta, fv, i * (m_featureSize + 1));
		return Utils.maxOfArrayIndex(m_cache);
	}
	
	protected void debug(_Doc d) {
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, true), "UTF-8"));
			_SparseFeature[] fv = d.getSparse();
			int fid, offset;
			double fvalue;		
			
			writer.write(d.toString());
			
			writer.write("\nBIAS");
			for(int k=0; k<m_classNo; k++) {
				offset = k * (m_featureSize + 1);
				writer.write(String.format("\t%.4f", m_beta[offset]));					
			}
			writer.write("\n");
			
			for(int i=0; i<fv.length; i++) {
				fid = fv[i].getIndex();
				if (fid>=m_corpus.getFeatureSize())
					break; // beyond text feature range
				fvalue = fv[i].getValue();				
				
				writer.write(m_corpus.getFeature(fid));
				for(int k=0; k<m_classNo; k++) {
					offset = k * (m_featureSize + 1) + fid + 1;
					writer.write(String.format("\t%.4f", fvalue*m_beta[offset]));					
				}
				writer.write("\n");
			}
			
			writer.write("Pred");
			for(int k=0; k<m_classNo; k++) 
				writer.write(String.format("\t%.4f", m_cache[k]));		
			writer.write("\n\n");
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelLocation), "UTF-8"));
			int offset, fSize = m_corpus.getFeatureSize();//does not include bias and time features
			for(int i=0; i<fSize; i++) {
				writer.write(m_corpus.getFeature(i));
				
				for(int k=0; k<m_classNo; k++) {
					offset = 1 + i + k * (m_featureSize + 1);//skip bias
					writer.write("\t" + m_beta[offset]);
				}
				writer.write("\n");
			}
			writer.close();
			
			System.out.format("%s is saved to %s\n", this.toString(), modelLocation);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
