/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with zero order gradient update, i.e., we will only touch the current user's gradient
 */
public class asyncCoLinAdapt extends CoLinAdapt {

	int[] m_userOrder; // visit order of different users during online learning
	
	public asyncCoLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		
		// all three test modes for asyncCoLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}

	public void setTestMode(TestMode mode) {
		m_testmode = mode;
	}
	
	@Override
	void constructNeighborhood() {
		super.constructNeighborhood();
		int adaptSize = 0;//total number of adaptation instances
		
		//construct the reverse link
		_CoLinAdaptStruct ui, uj;
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)(m_userList.get(i));
			for(_RankItem nit:ui.getNeighbors()) {
				uj = (_CoLinAdaptStruct)(m_userList.get(nit.m_index));//uj is a neighbor of ui
				
				uj.addReverseNeighbor(i, nit.m_value);
			}
			adaptSize += ui.getAdaptationSize();
		}
		
		//construct the order of online updating
		ArrayList<_RankItem> userorder = new ArrayList<_RankItem>();
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)(m_userList.get(i));
			
			for(_Review r:ui.getReviews()) {//reviews in each user is already ordered by time
				if (r.getType() == rType.ADAPTATION) {
					userorder.add(new _RankItem(i, r.getTimeStamp()));//to be in ascending order
				}
			}
		}
		
		Collections.sort(userorder);
		
		m_userOrder = new int[adaptSize];
		for(int i=0; i<adaptSize; i++)
			m_userOrder[i] = userorder.get(i).m_index;
	}
	
	@Override
	protected void gradientByFunc(_LinAdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review);
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
		double coef = 2 * sim, dA, dB;
		int offset = m_dim*2*ui.m_id;
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offset + k] += dA;
			m_g[offset + k + m_dim] += dB;
		}
	}
	
	protected double gradientTest(_CoLinAdaptStruct user) {
		int offset, uid = 2*m_dim*user.m_id;
		double magA = 0, magB = 0;
		
		for(int i=0; i<m_dim; i++){
			offset = uid + i;
			magA += m_g[offset]*m_g[offset];
			magB += m_g[offset+m_dim]*m_g[offset+m_dim];
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 0.50, gNorm, gNormOld = Double.MAX_VALUE;
		int updateCount = 0;
		_CoLinAdaptStruct user;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		
		initLBFGS();
		for(int t=0; t<m_userOrder.length; t++) {
			user = (_CoLinAdaptStruct)m_userList.get(m_userOrder[t]);

			if(user.hasNextAdaptationIns()) {
				// test the latest model
				if (m_testmode!=TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
					perfStat = user.getPerfStat();
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					perfStat.addOnePredResult(predL, trueL);
				} // in batch mode we will not accumulate the performance during adaptation			
				
				// prepare to adapt: initialize gradient	
				Arrays.fill(m_g, 0); 
				calculateGradients(user);
				gNorm = gradientTest(user);
				
				if (m_displayLv==1) {
					if (gNorm<gNormOld)
						System.out.print("o");
					else
						System.out.print("x");
				}
				
				//gradient descent
				gradientDescent(user, initStepSize/(1.0+user.getAdaptedCount()));
				gNormOld = gNorm;
				
				if (m_displayLv>0 && ++updateCount%100==0)
					System.out.println();
			}			
		}
		
		if (m_displayLv>0)
			System.out.println();
		
		for(_LinAdaptStruct u:m_userList)
			setPersonalizedModel(u);
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
