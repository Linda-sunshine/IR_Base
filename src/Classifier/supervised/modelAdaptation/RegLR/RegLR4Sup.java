package Classifier.supervised.modelAdaptation.RegLR;

import java.util.Arrays;
import java.util.HashMap;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class RegLR4Sup extends RegLR {
	
	//shared space for LBFGS optimization, dim = m_featureSize + 1
	protected double[] m_w; //weights for all users/super user.
	public RegLR4Sup(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		// default value of trade-off parameters
		m_eta1 = 0.5;
		
		// the only test mode for RegLR is batch
		m_testmode = TestMode.TM_batch;
	}
	
	@Override
	public void initLBFGS(){
		if(m_g == null)
			m_g = new double[m_featureSize + 1];
		if(m_diag == null)
			m_diag = new double[m_featureSize + 1];
		m_w = new double[m_featureSize + 1];
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_AdaptStruct user){
		return calcLogLikelihood(user); //log likelihood.
	}
	
//	@Override
//	//Calculate the function value of the new added instance.
//	protected double calcLogLikelihood(_AdaptStruct user){
//		double L = 0; //log likelihood.
//		double Pi = 0;
//		
//		for(_Review review:user.getReviews()){
//			if (review.getType() != rType.ADAPTATION)
//				continue; // only touch the adaptation data
//			
//			Pi = logit(review.getSparse(), user);
//			if(review.getYLabel() == 1) {
//				if (Pi>0.0)
//					L += Math.log(Pi);					
//				else
//					L -= Utils.MAX_VALUE;
//			} else {
//				if (Pi<1.0)
//					L += Math.log(1 - Pi);					
//				else
//					L -= Utils.MAX_VALUE;
//			}
//		}
//		return L;
//	}
	
	public double calculateR1(){
		_AdaptStruct user = m_userList.get(0);
		double R1 = 0;
		//Add regularization parts.
		for(int i=0; i<m_featureSize+1; i++)
			R1 += (user.getPWeight(i) - m_gWeights[i]) * (user.getPWeight(i) - m_gWeights[i]);//(w^u_i-w^g_i)^2
		return R1;
	}
	
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void calculateGradients(_AdaptStruct user){
		gradientByFunc(user);
	}
	@Override
	protected void gradientByFunc(_AdaptStruct user, _Doc review, double weight) {
		int n; // feature index
		double delta = (review.getYLabel() - logit(review.getSparse(), user))/getAdaptationSize(user);
		if(!m_LNormFlag)
			delta *= getAdaptationSize(user);
		
		//Bias term.
		m_g[0] -= weight*delta; //a[0] = w0*x0; x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[n] -= weight * delta * fv.getValue();
		}
	}
	
	protected void gradientByR1(){
		_AdaptStruct user = m_userList.get(0);
		//R1 regularization part
		for(int k=0; k<m_featureSize+1; k++)
			m_g[k] += 2 * m_eta1 * (user.getPWeight(k) - m_gWeights[k]);// add 2*eta1*(w^u_k-w^g_k)
	}
	
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, L, displayCount = 0, oldFValue = Double.MAX_VALUE;
		
		init();
		initLBFGS();
		try{	
			do{
				L = 0;
				Arrays.fill(m_g, 0); // initialize gradient					
				for(_AdaptStruct user:m_userList) {			
					L += calculateFuncValue(user);
					calculateGradients(user);
				}
				fValue = m_eta1 * calculateR1() - L;
				gradientByR1();
				if (m_displayLv==2) {
					System.out.println("Fvalue is " + fValue);
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					if (++displayCount%100==0)
						System.out.println();
				} 
				oldFValue = fValue;
					
				LBFGS.lbfgs(m_w.length, 6, m_w, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				setPersonalizedModel();
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
		return oldFValue;
	}
	
	@Override
	protected void setPersonalizedModel() {
		for(_AdaptStruct u: m_userList){
			u.setPersonalizedModel(m_w);
		}
	}
}
