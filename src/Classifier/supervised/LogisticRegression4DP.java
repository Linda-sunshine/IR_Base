package Classifier.supervised;

import java.util.Collection;

import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class LogisticRegression4DP extends LogisticRegression {	
	
	public LogisticRegression4DP(int classNo, int featureSize, double lambda){
		super(classNo, featureSize, lambda);
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
			Yi = doc.getClusterNo();
			fv = doc.getLMSparse();
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
	
	public double[] calcCProbs(_Doc doc){
		_SparseFeature[] fv = doc.getLMSparse();
		double[] cProbs = new double[m_classNo];
		for(int i = 0; i < m_classNo; i++)
			cProbs[i] = 1/(1+ Math.exp(-Utils.dotProduct(m_beta, fv, i * (m_featureSize + 1))));
		Utils.L1Normalization(cProbs);
		return cProbs;
	}
}
