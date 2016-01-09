/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._PerformanceStat;
import structures._Review;
import structures._PerformanceStat.TestMode;
import utils.Utils;

/**
 * @author Hongning Wang
 * online learning of LinAdapt
 */
public class asyncLinAdapt extends LinAdapt {

	public asyncLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		
		// all three test modes for asyncLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}
	
	public void setTestMode(TestMode mode) {
		m_testmode = mode;
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 0.50, A[], gNorm, gNormOld = Double.MAX_VALUE;;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		
		initLBFGS();
		for(_LinAdaptStruct user:m_userList) {
			A = user.getA();
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
				Utils.add2Array(A, m_g, -initStepSize/(1.0+user.getAdaptedCount()));
				gNormOld = gNorm;
			}
			
			if (m_displayLv>0)
				System.out.println();
			setPersonalizedModel(user);
		}
	}
	
	@Override
	protected int getAdaptationSize(_LinAdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	@Override
	protected void gradientByFunc(_LinAdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0);//equal weight for the user's own adaptation data
	}
}
