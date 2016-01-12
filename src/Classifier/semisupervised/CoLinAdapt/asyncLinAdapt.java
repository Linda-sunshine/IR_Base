/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
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
	
	static double getStepSize(double initStepSize, _LinAdaptStruct user) {
		return (0.1+0.9*Math.random()) * initStepSize/(2.0+user.getUpdateCount());
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 0.50, gNorm, gNormOld = Double.MAX_VALUE;;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		
		initLBFGS();
		init();
		for(_LinAdaptStruct user:m_userList) {
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
				gradientDescent(user, initStepSize);
				gNormOld = gNorm;
			}
			
			if (m_displayLv>0)
				System.out.println();
			setPersonalizedModel(user);
		}
	}
	
	// update this current user only
	void gradientDescent(_LinAdaptStruct user, double initStepSize) {
		double stepSize = asyncLinAdapt.getStepSize(initStepSize, user);
		Utils.add2Array(user.getA(), m_g, -stepSize);
		user.incUpdatedCount(1.0);
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
