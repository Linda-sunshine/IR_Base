/**
 * 
 */
package Ranker;

import java.util.Arrays;
import java.util.Collection;

import cern.jet.random.tdouble.Normal;

import utils.Utils;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

/**
 * @author hongning
 * Logistic Regression for one-class classification: essentially this is RankNet
 */
public class RankNet {

	double[] m_beta;
	double[] m_g, m_diag;
	double m_lambda;
	
	int[] m_signs; // sign of feature weights during random initialization
	
	public RankNet(int fSize, double lambda) {
		m_beta = new double[fSize]; //Initialization, no bias term
		m_g = new double[m_beta.length];
		m_diag = new double[m_beta.length];
		m_lambda = lambda;
	}	

	@Override
	public String toString() {
		return String.format("RankNet[F:%d, L:%.2f]", m_beta.length, m_lambda);
	}
	
	protected void init() {		
		Arrays.fill(m_diag, 0);
		
		double lambda = 1.0/Math.sqrt(m_lambda);
		for(int i=0; i<m_beta.length; i++)
			m_beta[i] = Normal.staticNextDouble(0, lambda);
		
		if (m_signs!=null) {//enforce sign over the initial setting of feature weights
			for(int i=0; i<m_beta.length; i++) {
				if (m_signs[i]*m_beta[i]<0)
					m_beta[i] = -m_beta[i];
			}
		}
	}
	
	public void setSigns(int[] signs) {
		m_signs = signs;
	}
	
	public double train(Collection<double[]> trainSet) {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0;
		int fSize = m_beta.length;
		
		init();
		try{
			do {
				fValue = calcFuncGradient(trainSet);
				LBFGS.lbfgs(fSize, 5, m_beta, fValue, m_g, false, m_diag, iprint, 8e-2, 1e-32, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
		return fValue;
	}
	
	//This function is used to calculate the value and gradient with the new beta.
	protected double calcFuncGradient(Collection<double[]> trainSet) {		
		double gValue = 0, fValue = 0, likelihood = 0;

		// Add the L2 regularization.
		double L2 = 0, b;
		for(int i = 0; i < m_beta.length; i++) {
			b = m_beta[i];
			m_g[i] = 2 * m_lambda * b;
			L2 += b * b;
		}
		
		for (double[] doc: trainSet) {
			//compute P(Y=1|X=xi)
			fValue = score(doc);
			gValue = fValue - 1.0;	
			likelihood += Math.log(fValue);
			
			for(int i=0; i<doc.length; i++)
				m_g[i] += gValue * doc[i];
		}
			
		// LBFGS is used to calculate the minimum value while we are trying to calculate the maximum likelihood.
		return m_lambda*L2 - likelihood;
	}
	
	public double score(double[] x) {
		return Utils.logistic(Utils.dotProduct(x, m_beta));
	}
	
	public double[] getWeights() {
		return m_beta;
	}
}
