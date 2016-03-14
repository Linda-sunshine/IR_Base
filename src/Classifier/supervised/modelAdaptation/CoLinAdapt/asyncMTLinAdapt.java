package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._PerformanceStat;
import structures._Review;
import structures._PerformanceStat.TestMode;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;

public class asyncMTLinAdapt extends MTLinAdaptWithSupUsr{

	double m_initStepSize = 0.50;
	
	public asyncMTLinAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}
	
	@Override
	public String toString() {
		return String.format("asyncMTLinAdapt[dim:%d,SupDim:%d, eta1:%.3f,eta2:%.3f, lambda1:%.3f. lambda2:%.3f]", m_dim, m_dimSup, m_eta1, m_eta2, m_lambda1, m_lambda2);
	}

	// We use the whole vector to store the gradient, which is shared among all users.
	@Override
	protected void initLBFGS(){ 
		int vSize = 2*m_dim*m_userList.size() + m_dimSup*2;
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
		gradientByRs();
	}
	
	//this is online training in each individual user
	@Override
	public double train(){
		double gNorm, gNormOld = Double.MAX_VALUE;;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		_CoLinAdaptStruct user;
		
		initLBFGS();
		init();
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			
			while(user.hasNextAdaptationIns()) {
				// test the latest model before model adaptation
				if (m_testmode != TestMode.TM_batch &&(doc = user.getLatestTestIns()) != null) {
					perfStat = user.getPerfStat();
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					perfStat.addOnePredResult(predL, trueL);
				} // in batch mode we will not accumulate the performance during adaptation				
				
				// prepare to adapt: initialize gradient	
				Arrays.fill(m_g, 0);
				calculateGradients(user);
				gNorm = gradientTest();
				
				if (m_displayLv==1) {
					if (gNorm<gNormOld)
						System.out.print("o");
					else
						System.out.print("x");
				}
				//gradient descent
				gradientDescent(user, m_initStepSize, 1.0);
				gNormOld = gNorm;
			}
			
			if (m_displayLv>0)
				System.out.println();
		}
		
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}	
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0);//equal weight for the user's own adaptation data
	}
	
	// update this current user only
	void gradientDescent(_CoLinAdaptStruct user, double initStepSize, double inc) {
		double a, b, stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = 2 * m_dim * user.getId();
		for (int k = 0; k < m_dim; k++) {
			a = user.getScaling(k) - stepSize * m_g[offset + k];
			user.setScaling(k, a);

			b = user.getShifting(k) - stepSize * m_g[offset + k + m_dim];
			user.setShifting(k, b);
		}
		user.incUpdatedCount(inc);
	}
}
