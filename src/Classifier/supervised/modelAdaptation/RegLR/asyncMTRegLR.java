package Classifier.supervised.modelAdaptation.RegLR;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._PerformanceStat.TestMode;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
/**
 * The modified version of MT-SVM since it cannot be performed in online mode.
 * @author lin
 */
public class asyncMTRegLR extends asyncRegLR{
	double m_u; //The parameter of the global part.
	double[] m_glbWeights; // The shared global weights.
	
	public asyncMTRegLR(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_u = 1;
	}
	
	public void setTradeOffParam(double u){
		m_u = Math.sqrt(u);
	}
	
	@Override
	public String toString() {
		return String.format("asyncMTRegLR[u:%.2f,initStepSize: %.3f, eta1:%.3f]", m_u, m_initStepSize, m_eta1);
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		super.loadUsers(userList);
		m_glbWeights = new double[m_featureSize+1];
	}
	
	@Override
	protected void initLBFGS(){
		// We need to consider global model as one part, we have (usersize + 1) in total.
		if(m_g == null)
			m_g = new double[(m_featureSize+1)*(m_userList.size()+1)];
		if(m_diag == null)
			m_diag = new double[m_g.length];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}

	// Every user is represetented by (u*global + individual)
	protected double logit(_SparseFeature[] fvs, _AdaptStruct user){
		// User bias.
		double sum = user.getPWeight(0);
		// Global bias.
		sum += m_u * m_glbWeights[0]; // bias term
		for(_SparseFeature f:fvs){
			// User features.
			sum += user.getPWeight(f.getIndex()+1) * f.getValue();	
			// Global features.
			sum += m_u * m_glbWeights[f.getIndex()+1]*f.getValue();
		}
		return Utils.logistic(sum);
	}
	
	protected void gradientByFunc(_AdaptStruct user, _Doc review, double weight) {
		int n; // feature index
		int glbOffset = (m_featureSize+1)*m_userList.size();
		int offset = (m_featureSize+1)*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[offset] -= weight*delta; //a[0] = w0*x0; x0=1
		m_g[glbOffset] -= weight*m_u*delta;// offset for the global part.
		
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= weight * delta * fv.getValue();// User part.
			m_g[glbOffset + n] -= weight * delta * m_u * fv.getValue(); // Global part.
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(_AdaptStruct user){
		int offset = (m_featureSize+1)*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int glbOffset = (m_featureSize+1)*m_userList.size();
		
		//R1 regularization part
		for(int k=0; k<m_featureSize+1; k++){
			m_g[offset + k] += 2 * m_eta1 * (user.getPWeight(k) - m_gWeights[k]);// add 2*eta1*(w^u_k-w^g_k)
			m_g[glbOffset + k] += 2 * m_eta1 * (m_u * m_glbWeights[k] - m_gWeights[k]);
		}
	}
	
	@Override
	public double train() {
		double gNorm, gNormOld = Double.MAX_VALUE;
		int predL, trueL;
		double val;
		_Review doc;
		_AdaptStruct user;
		_PerformanceStat perfStat;

		initLBFGS();
		init();
		try{
			m_writer = new PrintWriter(new File(String.format("%s_online_MTRegLR.txt", m_dataset)));
			for(int i=0; i<m_userList.size(); i++) {
				user = m_userList.get(i);
			
				while(user.hasNextAdaptationIns()) {
					// test the latest model before model adaptation
					if (m_testmode != TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
						perfStat = user.getPerfStat();						
						val = logit(doc.getSparse(), user);
						predL = predict(doc, user);
						trueL = doc.getYLabel();
						perfStat.addOnePredResult(predL, trueL);
						m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
					} // in batch mode we will not accumulate the performance during adaptation				
				
					gradientDescent(user, m_initStepSize, 1.0);
				
					//test the gradient only when we want to debug
					if (m_displayLv>0) {
						gNorm = gradientTest();				
						if (m_displayLv==1) {
							if (gNorm<gNormOld)
								System.out.print("o");
							else
								System.out.print("x");
						}				
						gNormOld = gNorm;
					}
				}
				m_writer.flush();
				if (m_displayLv==1)
					System.out.println();
			}
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}
	
	// update this current user only
	public void gradientDescent(_AdaptStruct user, double initStepSize, double inc) {
		double stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int glbOffset = (m_featureSize+1)*m_userList.size();
		int offset = (m_featureSize+1) * user.getId();
		
		double[] g;
		
		//get gradient
		Arrays.fill(m_g, 0);
		calculateGradients(user);
	
		//update the individual user
		g = Arrays.copyOfRange(m_g, offset, offset+m_featureSize+1);
		Utils.add2Array(user.getUserModel(), g, -stepSize);
		
		//update the shared global part.
		g = Arrays.copyOfRange(m_g, glbOffset, (m_featureSize+1)*(m_userList.size()+1));
		Utils.add2Array(m_glbWeights, g, -stepSize);
		
		//update the record of updating history
		user.incUpdatedCount(inc);
	}
}
