package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._SparseFeature;
import structures._User;

public class MTLinAdapt extends CoLinAdapt {

	double[] m_A; // [A_0, A_1, A_2,..A_s]Transformation matrix shared by super user and individual users.

	double[] m_sWeights; // Weights for the super user.
	double m_lambda1; // Scaling coefficient for R^1(A_s)
	double m_lambda2; // Shifting coefficient for R^1(A_s)
	boolean m_LNormFlag; // Decide if we will normalize the likelihood.
	
	public MTLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
						int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		m_lambda1 = 0.5;
		m_lambda2 = 1;
		m_LNormFlag = true;
	}
	
	public void setLNormFlag(boolean b){
		m_LNormFlag = b;
	}
	
	public void setRsTradeOffs(double lmd1, double lmd2){
		m_lambda1 = lmd1;
		m_lambda2 = lmd2;
	}
	
	@Override
	public String toString() {
		return String.format("MT-LinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,lambda1:%.3f,lambda2:%.3f,k:%d,NB:%s]", 
				m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_lambda1, m_lambda2, m_topK, m_sType);
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
		_CoLinAdaptStruct.sharedA = new double[vSize*(m_userList.size()+1)];
		//pass the reference of shared A to the algorithm.
		m_A = _CoLinAdaptStruct.sharedA;
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
		// Init A_s with [1,1,1,..,0,0,0,...].
		for(int i=m_userList.size()*m_dim*2; i<m_userList.size()*m_dim*2+m_dim; i++)
			m_A[i] = 1;
		
		// Init m_sWeights with global weights;
		m_sWeights = new double[m_featureSize + 1];
		m_sWeights = Arrays.copyOfRange(m_gWeights, 0, m_gWeights.length);
	}
	
	@Override
	protected void initLBFGS(){
		int vSize = 2*m_dim*(m_userList.size()+1);
		
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
	
	//Calculate the function value of the new added instance.
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		double L = calcLogLikelihood(ui); //log likelihood.
		if(!m_LNormFlag)
			L *= ui.getAdaptationSize();
		
		//Add regularization parts.
		double R1 = 0;
		for(int k=0; k<m_dim; k++){
			R1 += m_eta1 * (ui.getScaling(k)-1) * (ui.getScaling(k)-1);//(a[i]-1)^2
			R1 += m_eta2 * ui.getShifting(k) * ui.getShifting(k);//b[i]^2
		}
		return R1 - L;
	}
	
	// Calculate the R1 for the super user, As.
	protected double calculateRs(){
		int offset = m_userList.size()*m_dim*2; // Access the As.
		double rs = 0;
		for(int i=0; i < m_dim; i++){
			rs += m_lambda1 * (m_A[offset + i] - 1) * (m_A[offset + i] - 1); // Get scaling of super user.
			rs += m_lambda2 * m_A[offset + i + m_dim] * m_A[offset + i + m_dim]; // Get shifting of super user.
		}
		return rs;
	}
	
	@Override
	// Since I cannot access the method in LinAdapt or in RegLR, I Have to rewrite.
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
	}
	
	// Gradients for the gs.
	protected void gradientByRs(){
		int offset = m_userList.size() * m_dim * 2;
		for(int i=0; i < m_dim; i++){
			m_g[offset + i] += 2 * m_lambda1 * (m_A[offset + i] - 1);
			m_g[offset + i + m_dim] += 2 * m_lambda2 * m_A[offset + i + m_dim];
		}
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
		m_g[offset] -= weight*delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		m_g[offset + m_dim] -= weight*delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= weight*delta*ui.getScaling(0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		m_g[offsetSup + m_dim] -= weight*delta*ui.getScaling(0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= weight*delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			m_g[offset + m_dim + k] -= weight*delta*fv.getValue(); // x_di
			
			m_g[offsetSup + k] -= weight*delta*ui.getScaling(k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			m_g[offsetSup + m_dim + k] -= weight*delta*ui.getScaling(k)*fv.getValue(); // a_i*x_di
		}
	}
	
	//this is batch training in each individual user
	@Override
	public double train() {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue, oldFValue = Double.MAX_VALUE;
		int vSize = 2 * m_dim * (m_userList.size()+1), displayCount = 0;
		double oldMag = 0;
		_LinAdaptStruct user;

		initLBFGS();
		init();
		try {
			do {
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient

				// accumulate function values and gradients from each user
				for (int i = 0; i < m_userList.size(); i++) {
					user = (_LinAdaptStruct) m_userList.get(i);
					fValue += calculateFuncValue(user); // L + R^1(A_i)
					calculateGradients(user);
				}
				// The contribution from R^1(A_s) to both function value and gradients.
				fValue += calculateRs(); // + R^1(A_s)
				gradientByRs(); // Gradient from R^1(A_s)
				
				// added by Lin for stopping lbfgs.
				double curMag = gradientTest();
//				if (Math.abs(oldMag - curMag) < 0.1)
//					break;
//				oldMag = curMag;

				if (m_displayLv == 2) {
					System.out.print("Fvalue is " + fValue);
				} else if (m_displayLv == 1) {
					if (fValue < oldFValue)
						System.out.print("o");
					else
						System.out.print("x");

					if (++displayCount % 100 == 0)
						System.out.println();
				}
				oldFValue = fValue;

				LBFGS.lbfgs(vSize, 5, m_A, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);// In the training process, A is updated.
			} while (iflag[0] != 0);
			System.out.println();
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}

		setPersonalizedModel();
		return 0;
//		return oldFValue;
	}
	
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid;
		_CoLinAdaptStruct ui;
		// Get a copy of super user's transformation matrix.
		double[] As = Arrays.copyOfRange(m_A, m_userList.size()*m_dim*2, (m_userList.size()+1)*m_dim*2);
		
		// Set the bias term for ws.
		m_sWeights[0] = As[0] * m_gWeights[0] + As[m_dim];
		// Set the other terms for ws.
		for(int n=0; n<m_featureSize; n++){
			gid = m_featureGroupMap[1+n];
			m_sWeights[n+1] = As[gid] * m_gWeights[1+n] + As[gid+ m_dim];
		}
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			
			//set bias term
//			m_pWeights[0] = ui.getScaling(0) * m_sWeights[0] + ui.getShifting(0);
			m_pWeights[0] = m_sWeights[0];

//			//set the other features
//			for(int n=0; n<m_featureSize; n++) {
//				gid = m_featureGroupMap[1+n];
//				m_pWeights[1+n] = ui.getScaling(gid) * m_sWeights[1+n] + ui.getShifting(gid);
//			}
			//set the other features
			for(int n=0; n<m_featureSize; n++) {
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
			value = m_A[offsetSup] * m_gWeights[0] + m_A[offsetSup + m_dim]; // Set the bias term for ws.
		else{
			// Set the other terms for ws.
			gid = m_featureGroupMap[index];
			value = m_A[offsetSup + gid] * m_gWeights[index] + m_A[offsetSup + gid + m_dim];
		}
		return value;
	}
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, uid;
		double magA = 0, magB = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}
		
		if (m_displayLv==2)
			System.out.format("\t mag: %.4f\n", magA + magB);
		return magA + magB;
	}
	
	public double[] getSupWeights(){
		return m_sWeights;
	}
	
	public double[] getGlobalWeights(){
		return m_gWeights;
	}

	public void printWeights(String path) throws FileNotFoundException{
		String light = String.format("%s_light.txt", path);
		String medium = String.format("%s_medium.txt", path);
		String heavy = String.format("%s_heavy.txt", path);
		PrintWriter writerLight = new PrintWriter(new File(light));
		PrintWriter writerMedium = new PrintWriter(new File(medium));
		PrintWriter writerHeavy = new PrintWriter(new File(heavy));
		PrintWriter writer;
		_CoLinAdaptStruct ui;
		int rvwSize = 0;
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			rvwSize = ui.getUser().getReviewSize();

			if(rvwSize <= 10)
				writer = writerLight;
			else if(rvwSize <= 50)
				writer = writerMedium;
			else
				writer = writerHeavy;
	
			for(int j=m_dim*2*i; j<m_dim*2*(i+1); j++)
				writer.write(m_A[j] + "\t");
			writer.write("\n");
		}
		writerLight.close();
		writerMedium.close();
		writerHeavy.close();
	}
//	/***When we do feature selection, we will group features and store them in file. 
//	 * The index is the index of features and the corresponding number is the group index number.***/
//	public void loadFeatureGroupMap(String filename){
//			
//			m_featureGroupMap = new int[5000 + 1]; //One more term for bias, bias->0.
//			m_dim = 0;
//			//Group index starts from 0, so add 1 for it.
//			for(int i=0; i<5000; i++) {
//				m_featureGroupMap[i+1] = i+1;
//				if (m_dim < m_featureGroupMap[i+1])
//					m_dim = m_featureGroupMap[i+1];
//			}
//			m_dim = 5001;
//			
//			System.out.format("[Info]Feature group size %d\n", m_dim);
//	}
}
