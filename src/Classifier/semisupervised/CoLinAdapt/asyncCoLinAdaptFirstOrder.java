/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.HashMap;

import structures._RankItem;
import structures._Review;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with first order gradient descent, i.e., we will touch both the current user's R2 and all immediately related users
 */
public class asyncCoLinAdaptFirstOrder extends asyncCoLinAdapt {
	
	double m_neighborsHistoryWeight; // used to reweight the gradient of historical observations from neighbors
	
	public asyncCoLinAdaptFirstOrder(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap, double neighborsHistoryWeight) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_neighborsHistoryWeight = neighborsHistoryWeight;
	}
	
	@Override
	protected void calculateGradients(_LinAdaptStruct user){
		super.calculateGradients(user);
		if (m_neighborsHistoryWeight>0)
			cachedGradientByNeighorsFunc(user, m_neighborsHistoryWeight);
		gradientByRelatedR1((_CoLinAdaptStruct)user);
	}
	 
	//Calculate the reweighted gradients from neighbors' historical observations
	protected void cachedGradientByNeighorsFunc(_LinAdaptStruct user, double weight){		
		_CoLinAdaptStruct uj, ui = (_CoLinAdaptStruct)user;

		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			for(_Review r:uj.getAdaptationCache())
				gradientByFunc(uj, r, weight);
		}

		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			for(_Review r:uj.getAdaptationCache())
				gradientByFunc(uj, r, weight);
		}
	}
	
	@Override
	void gradientByR2(_CoLinAdaptStruct ui, _CoLinAdaptStruct uj, double sim) {
		double coef = 2 * sim, dA, dB;
		int offseti = m_dim*2*ui.m_id, offsetj = m_dim*2*uj.m_id;
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offseti + k] += dA;
			m_g[offseti + k + m_dim] += dB;
			
			// update uj's gradient
			m_g[offsetj + k] -= dA;
			m_g[offsetj + k + m_dim] -= dB;
		}
	}
	
	//compute gradient for all related user in first order connection (exclude itself)
	void gradientByRelatedR1(_CoLinAdaptStruct ui) {
		_CoLinAdaptStruct uj;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR1(uj);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR1(uj);
		}
	}

	@Override
	void gradientDescent(_CoLinAdaptStruct ui, double initStepSize, double inc) {
		super.gradientDescent(ui, initStepSize, inc);//update the current user
		
		_CoLinAdaptStruct uj;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, initStepSize, inc/3);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, initStepSize, inc/3);
		}
	}
}
