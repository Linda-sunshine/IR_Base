/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;

/**
 * @author Hongning Wang
 * online learning of LinAdapt
 */
public class asyncLinAdapt extends LinAdapt {
	double m_initStepSize = 0.50;
	String m_dataset;
	public asyncLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		
		// all three test modes for asyncLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}
	
	@Override
	public String toString() {
		return String.format("asyncLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_eta1, m_eta2);
	}
	
	public void setInitStepSize(double initStepSize) {
		m_initStepSize = initStepSize;
	}
	
	public void setDataset(String data){
		m_dataset = data;
	}
	
	//this is online training in each individual user
	@Override
	public double train(){
		double gNorm, gNormOld = Double.MAX_VALUE;;
		int predL, trueL;
		_Review doc;
		double val = 0;
		_PerformanceStat perfStat;
		_LinAdaptStruct user;
		
		initLBFGS();
		init();
		try{
			m_writer = new PrintWriter(new File(String.format("%s_online_LinAdapt.txt", m_dataset)));
			for(int i=0; i<m_userList.size(); i++) {
				user = (_LinAdaptStruct)m_userList.get(i);
				while(user.hasNextAdaptationIns()) {
					// test the latest model before model adaptation
					if (m_testmode != TestMode.TM_batch &&(doc = user.getLatestTestIns()) != null) {
						perfStat = user.getPerfStat();						
						val = logit(doc.getSparse(), user);
						predL = val>0.5?1:0;
						trueL = doc.getYLabel();
						perfStat.addOnePredResult(predL, trueL);
						m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
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
					asyncRegLR.gradientDescent(user, m_initStepSize, m_g);
					gNormOld = gNorm;
				}
			
				if (m_displayLv>0)
					System.out.println();
			}
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
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
}
