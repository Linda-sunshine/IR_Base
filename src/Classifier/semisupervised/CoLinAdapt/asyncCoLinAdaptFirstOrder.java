/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import structures._RankItem;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with first order gradient descent, i.e., we will touch both the current user's R2 and all immediately related users
 */
public class asyncCoLinAdaptFirstOrder extends asyncCoLinAdapt {

	public asyncCoLinAdaptFirstOrder(int classNo, int featureSize, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, topK, globalModel, featureGroupMap);
	}
	
	@Override
	protected void calculateGradients(_LinAdaptStruct user){
		super.calculateGradients(user);
		gradientByR2(user);
		gradientByRelatedR1((_CoLinAdaptStruct)user);
	}

	@Override
	void gradientByR2(_CoLinAdaptStruct ui, _CoLinAdaptStruct uj, double sim) {
		double coef = 2 * sim * m_eta3, dA, dB;
		int offset = m_dim*2;
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offset*ui.m_id + k] += dA;
			m_g[offset*ui.m_id + k + m_dim] += dB;
			
			// update uj's gradient
			m_g[offset*uj.m_id + k] -= dA;
			m_g[offset*uj.m_id + k + m_dim] -= dB;
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
	void gradientDescent(_CoLinAdaptStruct ui, double stepSize) {
		super.gradientDescent(ui, stepSize);
		
		_CoLinAdaptStruct uj;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, stepSize);//how should we pick the step size?
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			super.gradientDescent(uj, stepSize);
		}

	}
}
