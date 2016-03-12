package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._SparseFeature;
import structures._User;
import Classifier.supervised.modelAdaptation._AdaptStruct;
/***
 * 
 * @author lin
 * In this class, super user is only represented by the weights, 
 * in logit function, personalized weights are represented as:
 * A_i(p*w_s+q*w_g)^T*x_d
 */
public class MTLinAdaptWithSupUserNoAdpt extends MTLinAdapt{

	protected double m_p; // The coefficient in front of w_s.
	protected double m_q; // The coefficient in front of w_g.
	protected double m_beta; // The coefficient in front of R1(w_s)
	
	public MTLinAdaptWithSupUserNoAdpt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_p = 1;
		m_q = 0;
		m_beta = 1;
	}
	
	public void setWsWgCoefficients(double p, double q){
		m_p = p;
		m_q = q;
	}
	
	public void setR14SupCoefficients(double beta){
		m_beta = beta;
	}
	
	@Override
	public String toString() {
		return String.format("MT-LinAdaptWithSupUserNoAdpt[dim:%d,eta1:%.3f,eta2:%.3f,p:%.3f,q:%.3f,beta: %.3f,personalized:%b]", 
				m_dim, m_eta1, m_eta2, m_p, m_q, m_beta, m_personalized);
	}
	@Override
	public void loadUsers(ArrayList<_User> userList){
		int vSize = 2*m_dim;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[vSize*m_userList.size() + m_featureSize + 1];
		//pass the reference of shared A to the algorithm.
		m_A = _CoLinAdaptStruct.sharedA;
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
		
		// Init m_sWeights with global weights; copy sWeights to the shared A.
		m_sWeights = new double[m_featureSize + 1];
		m_sWeights = Arrays.copyOfRange(m_gWeights, 0, m_gWeights.length);
		System.arraycopy(m_sWeights, 0, m_A, vSize*m_userList.size(), m_sWeights.length);
	}
	
	@Override
	protected void initLBFGS(){
		// General users and super user have different dimensions.
		int vSize = 2*m_dim*m_userList.size() + m_featureSize + 1;
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	// We can do A_i*(m_p*w_s + m_q*w_g)^T*x_d at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int n = 0, k = 0; // feature index and feature group index
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		double value = ui.getScaling(0)*(m_p * getSupUserWeights(0) + m_q * m_gWeights[0]) + ui.getShifting(0);//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (ui.getScaling(k)*(m_p * getSupUserWeights(n) + m_q * m_gWeights[n]) + ui.getShifting(k)) * fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	public double getSupUserWeights(int index){
		return m_A[m_dim*2*m_userList.size() + index];
	}
	
	// Calculate the R1 for the super user, As.
	protected double calculateRs(){
		double rs = 0;
		for(int i=0; i < m_sWeights.length; i++)
			rs += getSupUserWeights(i)*getSupUserWeights(i);
		return rs * m_beta;
	}
	
	// Gradients from loglikelihood, contributes to both individual user's gradients and super user's gradients.
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		int n, k; // feature index and feature group index		
		int offset = 2*m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int offsetSup = 2*m_dim*m_userList.size();
		double delta;
		if(m_LNormFlag)
			delta = (review.getYLabel() - logit(review.getSparse(), ui)) / getAdaptationSize(ui);
		else
			delta = (review.getYLabel() - logit(review.getSparse(), ui));

		// Bias term for individual user.
		m_g[offset] -= weight*delta*(m_p * getSupUserWeights(0) + m_q * m_gWeights[0]); //a[0] = (p*w_s0+q*w_g0)*x0; x0=1
		m_g[offset + m_dim] -= weight*delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= weight*delta*ui.getScaling(0)*m_p; //a_s[0] = a_i0*p*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= weight*delta*(m_p * getSupUserWeights(n) + m_q * m_gWeights[n])*fv.getValue(); // (p*w_si+q*w_gi)*x_di
			m_g[offset + m_dim + k] -= weight*delta*fv.getValue(); // x_di
			
			m_g[offsetSup + n] -= weight*delta*ui.getScaling(k)*m_p*fv.getValue(); // a_i*p*x_di
		}
	}
	
	// Gradients for the gs.
	protected void gradientByRs(){
		int offset = m_userList.size() * m_dim * 2;
		for(int i=0; i < m_sWeights.length; i++)
			m_g[offset + i] += 2 * m_beta * getSupUserWeights(i);
	}
	
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid;
		_CoLinAdaptStruct ui;
		
		// Get a copy of super user's weightsa.
		m_sWeights = Arrays.copyOfRange(m_A, m_userList.size()*m_dim*2, m_A.length);
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			
			if(m_personalized){
				//set bias term
				m_pWeights[0] = ui.getScaling(0)*(m_p*m_sWeights[0]+m_q*m_gWeights[0]) + ui.getShifting(0);
				//set the other features
				for(int n=0; n<m_featureSize; n++) {
					gid = m_featureGroupMap[1+n];
					m_pWeights[1+n] = ui.getScaling(gid) * (m_p*m_sWeights[1+n]+m_q*m_gWeights[1+n]) + ui.getShifting(gid);
				}
			} else{ // Set super user == general user.
				m_pWeights[0] = m_sWeights[0];
				for(int n=0; n<m_featureSize; n++)
					m_pWeights[1+n] = m_sWeights[1+n];
			}
			ui.setPersonalizedModel(m_pWeights);
		}
	}
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, offsetSup, uid;
		double magA = 0, magB = 0, magS = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}

		offsetSup = vSize * m_userList.size();
		for(int i=0; i<m_sWeights.length; i++)
			magS += m_g[offsetSup+i] * m_g[offsetSup+i];
		
		if (m_displayLv==2)
			System.out.format("\t mag: %.4f\n", magA + magB + magS);
		return magA + magB + magS;
	}
}
