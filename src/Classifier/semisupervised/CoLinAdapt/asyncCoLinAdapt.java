/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Arrays;

import structures._RankItem;
import structures._Review;
import utils.Utils;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with zero order gradient update, i.e., we will only touch the current user's gradient
 */
public abstract class asyncCoLinAdapt extends CoLinAdapt {

	int[] m_userOrder; // visit order of different users during online learning
	
	public asyncCoLinAdapt(int classNo, int featureSize, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, topK, globalModel, featureGroupMap);
	}

	@Override
	void constructNeighborhood() {
		super.constructNeighborhood();
		int adaptSize = 0, aPtr = 0;//total number of adaptation instances
		
		//construct the reverse link
		_CoLinAdaptStruct ui, uj;
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)(m_userList.get(i));
			for(_RankItem nit:ui.getNeighbors()) {
				uj = (_CoLinAdaptStruct)(m_userList.get(nit.m_index));
				
				uj.addReverseNeighbor(i, nit.m_value);
			}
			adaptSize += ui.getAdaptationSize();
		}
		
		//construct the order of online updating
		m_userOrder = new int[adaptSize];
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)(m_userList.get(i));
			for(int j=0; j<ui.getAdaptationSize(); j++) {
				m_userOrder[aPtr] = i;
				aPtr ++;
			}
		}
		
		Utils.shuffle(m_userOrder, adaptSize);//using random order?
	}
	
	@Override
	protected void gradientByFunc(_LinAdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review);
	}
	
	protected double gradientTest(_CoLinAdaptStruct user) {
		int offset, uid = 2*m_dim*user.m_id;
		double magA = 0, magB = 0;
		
		for(int i=0; i<m_dim; i++){
			offset = uid + i;
			magA += m_g[offset]*m_g[offset];
			magB += m_g[offset+m_dim]*m_g[offset+m_dim];
		}
	
		System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 1.0;
		_CoLinAdaptStruct user;
		
		initLBFGS();
		for(int t=0; t<m_userOrder.length; t++) {
			user = (_CoLinAdaptStruct)m_userList.get(m_userOrder[t]);

			while(user.hasNextAdaptationIns()) {
				Arrays.fill(m_g, 0); // initialize gradient	
				calculateGradients(user);
				gradientTest(user);
				
				//gradient descent
				gradientDescent(user, initStepSize/(1.0+user.getAdaptedCount()));
			}
			setPersonalizedModel(user);
		}
	}
	
	@Override
	protected void gradientByR2(_LinAdaptStruct user){		
		_CoLinAdaptStruct uj, ui = (_CoLinAdaptStruct)user;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
	}
	
	void gradientByR2(_CoLinAdaptStruct ui, _CoLinAdaptStruct uj, double sim) {
		double coef = 2 * sim * m_eta3, dA, dB;
		int offset = m_dim*2;
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offset*ui.m_id + k] += dA;
			m_g[offset*ui.m_id + k + m_dim] += dB;
		}
	}
	
	// update this current user
	void gradientDescent(_CoLinAdaptStruct user, double stepSize) {
		double a, b;
		int offset = 2*m_dim*user.m_id;
		for(int k=0; k<m_dim; k++) {
			a = user.getScaling(k) - stepSize * m_g[offset + k];
			user.setScaling(k, a);
			
			b = user.getShifting(k) - stepSize * m_g[offset + k + m_dim];
			user.setShifting(k, b);
		}
	}	
}
