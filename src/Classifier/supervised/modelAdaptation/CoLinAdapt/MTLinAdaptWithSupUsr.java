package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._SparseFeature;
import structures._User;
import Classifier.supervised.modelAdaptation._AdaptStruct;

// In this class, we assign different dimension to the super user.
public class MTLinAdaptWithSupUsr extends MTLinAdapt {

	int m_dimSup;
	int[] m_featureGroupMap4SupUsr; // bias term is at position 0

	public MTLinAdaptWithSupUsr(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_dimSup = 0;
	}

	@Override
	public String toString() {
		return String.format("MT-LinAdaptWithSupUsr[dim:%d,supDime:%d, eta1:%.3f,eta2:%.3f,lambda1:%.3f,lambda2:%.3f, personalized: %b]", 
				m_dim, m_dimSup, m_eta1, m_eta2, m_lambda1, m_lambda2, m_personalized);
	}
	
	// Set the dimension of super user.
	public void setDimSup(int d){
		m_dimSup = d;
	}
	
	// Feature group map for the super user.
	public void loadFeatureGroupMap4SupUsr(String filename){
		
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimSup = m_featureSize + 1;
			m_featureGroupMap4SupUsr = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<m_featureSize; i++)
				m_featureGroupMap4SupUsr[i+1] = i+1;
			return;
		}
		
		// If there is feature grouping for the super user, load it.
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");//Group information of each feature.
			reader.close();
			
			m_featureGroupMap4SupUsr = new int[features.length + 1]; //One more term for bias, bias->0.
			m_dimSup = 0;
			//Group index starts from 0, so add 1 for it.
			for(int i=0; i<features.length; i++) {
				m_featureGroupMap4SupUsr[i+1] = Integer.valueOf(features[i]) + 1;
				if (m_dimSup < m_featureGroupMap4SupUsr[i+1])
					m_dimSup = m_featureGroupMap4SupUsr[i+1];
			}
			m_dimSup ++;
			
			System.out.format("[Info]Feature group size for super user %d\n", m_dimSup);
		} catch(IOException e){
			System.err.format("[Error]Fail to open super user group file %s.\n", filename);
		}
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
		_CoLinAdaptStruct.sharedA = new double[vSize*m_userList.size()+m_dimSup*2];
		//pass the reference of shared A to the algorithm.
		m_A = _CoLinAdaptStruct.sharedA;
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
		// Init A_s with [1,1,1,..,0,0,0,...].
		for(int i=m_userList.size()*m_dim*2; i<m_userList.size()*m_dim*2+m_dimSup; i++)
			m_A[i] = 1;
		
		// Init m_sWeights with global weights;
		m_sWeights = new double[m_featureSize + 1];
		m_sWeights = Arrays.copyOfRange(m_gWeights, 0, m_gWeights.length);
	}
	
	@Override
	protected void initLBFGS(){
		// General users and super user have different dimensions.
		int vSize = 2*m_dim*m_userList.size() + 2*m_dimSup;
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	// We can do A_i*A_s*w_g*x at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int n = 0, k = 0; // feature index and feature group index
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		double value = ui.getScaling(0)*getSupWeights(0) + ui.getShifting(0);//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (ui.getScaling(k)*getSupWeights(n) + ui.getShifting(k)) * fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	// Calculate the R1 for the super user, As.
	protected double calculateRs(){
		int offset = m_userList.size()*m_dim*2; // Access the As.
		double rs = 0;
		for(int i=0; i < m_dimSup; i++){
			rs += m_lambda1 * (m_A[offset + i] - 1) * (m_A[offset + i] - 1); // Get scaling of super user.
			rs += m_lambda2 * m_A[offset + i + m_dimSup] * m_A[offset + i + m_dimSup]; // Get shifting of super user.
		}
		return rs;
	}
	
	// Gradients for the gs.
	protected void gradientByRs(){
		int offset = m_userList.size() * m_dim * 2;
		for(int i=0; i < m_dimSup; i++){
			m_g[offset + i] += 2 * m_lambda1 * (m_A[offset + i] - 1);
			m_g[offset + i + m_dimSup] += 2 * m_lambda2 * m_A[offset + i + m_dimSup];
		}
	}
	
	// Gradients from loglikelihood, contributes to both individual user's gradients and super user's gradients.
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		
		int n, k, s; // feature index and feature group index		
		int offset = 2*m_dim*ui.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int offsetSup = 2*m_dim*m_userList.size();
		double delta;
		if(m_LNormFlag)
			delta = (review.getYLabel() - logit(review.getSparse(), ui)) / getAdaptationSize(ui);
		else
			delta = (review.getYLabel() - logit(review.getSparse(), ui));

		// Bias term for individual user.
		m_g[offset] -= weight*delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		m_g[offset + m_dim] -= weight*delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= weight*delta*ui.getScaling(0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		m_g[offsetSup + m_dimSup] -= weight*delta*ui.getScaling(0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= weight*delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			m_g[offset + m_dim + k] -= weight*delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			m_g[offsetSup + s] -= weight*delta*ui.getScaling(k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			m_g[offsetSup + m_dimSup + s] -= weight*delta*ui.getScaling(k)*fv.getValue(); // a_i*x_di
		}
	}
	
	@Override
	public int calcVSize(){
		return 2 * m_dim * m_userList.size() + m_dimSup * 2;
	}
	
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid;
		_CoLinAdaptStruct ui;
		// Get a copy of super user's transformation matrix.
		double[] As = Arrays.copyOfRange(m_A, m_userList.size()*m_dim*2, m_userList.size()*m_dim*2 + m_dimSup*2);
		
		// Set the bias term for ws.
		m_sWeights[0] = As[0] * m_gWeights[0] + As[m_dimSup];
		// Set the other terms for ws.
		for(int n=0; n<m_featureSize; n++){
			gid = m_featureGroupMap4SupUsr[1+n];
			m_sWeights[n+1] = As[gid] * m_gWeights[1+n] + As[gid+ m_dimSup];
		}
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			
			if(m_personalized){
				//set bias term
				m_pWeights[0] = ui.getScaling(0) * m_sWeights[0] + ui.getShifting(0);
				//set the other features
				for(int n=0; n<m_featureSize; n++) {
					gid = m_featureGroupMap[1+n];
					m_pWeights[1+n] = ui.getScaling(gid) * m_sWeights[1+n] + ui.getShifting(gid);
				}
			} else{ // Set super user == general user.
				m_pWeights[0] = m_sWeights[0];
				for(int n=0; n<m_featureSize; n++)
					m_pWeights[1+n] = m_sWeights[1+n];
			}
			ui.setPersonalizedModel(m_pWeights);
		}
	}
	
	// w_s = A_s * w_g
	public double getSupWeights(int index){
		int gid, offsetSup = m_userList.size() * 2 * m_dim;
		double value = 0;
		
		if(index == 0)
			value = m_A[offsetSup] * m_gWeights[0] + m_A[offsetSup + m_dimSup]; // Set the bias term for ws.
		else{
			// Set the other terms for ws.
			gid = m_featureGroupMap4SupUsr[index];
			value = m_A[offsetSup + gid] * m_gWeights[index] + m_A[offsetSup + gid + m_dimSup];
		}
		return value;
	}
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, offsetSup, uid;
		double magA = 0, magB = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}

		offsetSup = vSize * m_userList.size();
		for(int i=0; i<m_dimSup; i++){
			magA += m_g[offsetSup] * m_g[offsetSup];
			magB += m_g[offsetSup+m_dimSup] * m_g[offsetSup + m_dimSup];
		}
		
		if (m_displayLv==2)
			System.out.format("\t mag: %.4f\n", magA + magB);
		return magA + magB;
	}
}
