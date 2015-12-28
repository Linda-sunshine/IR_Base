/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Arrays;

import structures._Review;
import utils.Utils;

/**
 * @author Hongning Wang
 * online learning of LinAdapt
 */
public class asyncLinAdapt extends LinAdapt {

	public asyncLinAdapt(int classNo, int featureSize, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, globalModel, featureGroupMap);
	}
	
	//this is online training in each individual user
	@Override
	public void train(){
		double initStepSize = 1.50, A[];
		
		initLBFGS();
		for(_LinAdaptStruct user:m_userList) {
			A = user.getA();
			while(user.hasNextAdaptationIns()) {
				Arrays.fill(m_g, 0); // initialize gradient	
				calculateGradients(user);
				gradientTest();
				
				//gradient descent
				Utils.add2Array(A, m_g, -initStepSize/(1.0+user.getAdaptedCount()));
			}
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
