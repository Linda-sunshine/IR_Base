package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class MTLinAdapt extends CoLinAdapt {

	static double[] m_A; // [A_0, A_1, A_2,..A_s]Transformation matrix shared by super user and individual users.
	static double[] m_g; // Gradients shared by super user and individual users.
	double[] m_sWeights; // Weights for the super user.
	double m_lambda1; // Scaling coefficient for R^1(A_s)
	double m_lambda2; // Shifting coefficient for R^1(A_s)
	
	public MTLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, 
						int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		
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
		m_A = new double[vSize*(m_userList.size()+1)];
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, m_A, vSize*i, vSize);
		}
		// Init A_s with global model.
		System.arraycopy(m_gWeights, 0, m_A, vSize*m_userList.size(), vSize);
	}
	
	@Override
	public void initLBFGS(){
		int vSize = 2*m_dim*(m_userList.size()+1);
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
//	@Override
//	public double logit(_SparseFeature[] fvs, _AdaptStruct user){
//		double sum = user.getPWeight(0); // bias term
//		for(_SparseFeature f:fvs) 
//			sum += user.getPWeight(f.getIndex()+1) * f.getValue();		
//		return Utils.logistic(sum);
//	}
	
	// Calculate the R1 for the super user, As.
	public double calculateRs(){
		int offset = m_userList.size()*m_dim*2; // Access the As.
		double rs = 0;
		for(int i=0; i < m_dim; i++){
			rs += (m_A[offset + i] - 1)*(m_A[offset + i] - 1); // Get scaling of super user.
			rs += m_A[offset + i + m_dim] * m_A[offset + i + m_dim]; // Get shifting of super user.
		}
		return rs;
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
					fValue += super.calculateFuncValue(user); // L + R^1(A_i)
					calculateGradients(user);
				}
				
				fValue += calculateRs(); // fValue + R^1(A_s)
				// added by Lin for stopping lbfgs.
				double curMag = gradientTest();
				if (Math.abs(oldMag - curMag) < 0.1)
					break;
				oldMag = curMag;

				if (m_displayLv == 2) {
					System.out.println("Fvalue is " + fValue);
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
		return oldFValue;
	}
	
	// We need m_dim*2 more dimensions to contain the super user's gradient.
	public void calculateGradients(){
		
	}
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid;
		_LinAdaptStruct user;
		// Get a copy of super user's transformation matrix.
		double[] As = Arrays.copyOfRange(m_A, m_userList.size()*m_dim*2, (m_userList.size()+1)*m_dim*2);
		m_sWeights = new double[m_dim*2];
		
		// Set the bias term for ws.
		m_sWeights[0] = As[0] * m_gWeights[0] + As[m_dim];
		// Set the other terms for ws.
		for(int n=0; n<m_featureSize; n++){
			gid = m_featureGroupMap[1+n];
			m_sWeights[n+1] = As[gid] * m_gWeights[1+n] + As[gid+ m_dim];
		}
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			user = (_LinAdaptStruct)m_userList.get(i);
			
			//set bias term
			m_pWeights[0] = user.getScaling(0) * m_sWeights[0] + user.getShifting(0);
			
			//set the other features
			for(int n=0; n<m_featureSize; n++) {
				gid = m_featureGroupMap[1+n];
				m_pWeights[1+n] = user.getScaling(gid) * m_sWeights[1+n] + user.getShifting(gid);
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
}
