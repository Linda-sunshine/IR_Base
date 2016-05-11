package Classifier.supervised.modelAdaptation;

import java.util.HashMap;
import utils.Utils;
import Classifier.supervised.modelAdaptation.CoLinAdapt.CoLinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt._CoLinAdaptStruct;

// In this class, we do R1 over the global weights, ||wi-wg||.
public class CoLinAdaptWithR1OverWeights extends CoLinAdapt{

	public CoLinAdaptWithR1OverWeights(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}

	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		// log likelihood.
		double L = calcLogLikelihood(ui);
		double R1 = 0, R2;
		
		// R1 regularization.
		for(int i=0; i<ui.getPWeights().length; i++){
			R1 += m_eta1 * Utils.EuclideanDistance(ui.getPWeights(), m_gWeights);
		}
		
		// R2 regularization.
		R2 = super.calculateR2(u);
		
		return R1 + R2 - L;
	}
	
	// Calculate the gradients for the use in LBFGS.
	@Override
	protected void gradientByR1(_AdaptStruct u){
		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
		double dA, dB;
		int k, offset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		
		//R1 regularization part
		for(int n=0; n<m_featureSize+1; n++){
			k = m_featureGroupMap[n];
			dA = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]) * m_gWeights[n];
			dB = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]);
			
			m_g[offset + k] += dA;
			m_g[offset + k] += dB;
		}
	}
}
