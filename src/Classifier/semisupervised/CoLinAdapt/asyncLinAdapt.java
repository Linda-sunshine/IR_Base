/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._Review;
import utils.Utils;

/**
 * @author Hongning Wang
 * online learning of LinAdapt
 */
public class asyncLinAdapt extends LinAdapt {

	public asyncLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 0.50, A[], gNorm, gNormOld = Double.MAX_VALUE;;
		
		initLBFGS();
		for(_LinAdaptStruct user:m_userList) {
			A = user.getA();
			while(user.hasNextAdaptationIns()) {
				Arrays.fill(m_g, 0); // initialize gradient	
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
			
			System.out.println();
			setPersonalizedModel(user);
		}
	}
	
	@Override
	protected void gradientByFunc(_LinAdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review);
	}
}
