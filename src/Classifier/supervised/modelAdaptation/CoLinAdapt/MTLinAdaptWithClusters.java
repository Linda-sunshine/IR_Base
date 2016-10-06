package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class MTLinAdaptWithClusters extends MTLinAdapt{
	int m_k; // k clusters.
	int[] m_userClusterIndex; // The index is user index, the value is corresponding cluster no.

	public MTLinAdaptWithClusters(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap, String featureGroup4Sup, int kmeans, int[] indexMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap, featureGroup4Sup);
		m_k = kmeans;
		m_userClusterIndex = indexMap;
	}
	
	
	//Calculate the function value of the new added instance.
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		double L = calcLogLikelihood(u); //log likelihood.
		return -L;
	}
	
	
	// Gradients from loglikelihood, contributes to both individual user's gradients and super user's gradients.
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		int offset = getIndividualOffset(ui);//general enough to accommodate both LinAdapt and CoLinAdapt
		int offsetSup = getSupOffset();
		int cIndex = m_userClusterIndex[ui.getId()];
		
		int n, k, s; // feature index and feature group index		
		double delta = weight*(review.getYLabel() - logit(review.getSparse(), ui));
		if(m_LNormFlag)
			delta /= getAdaptationSize(ui);

		// Bias term for individual user.
		m_g[offset] -= delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		m_g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= delta*ui.getScaling(cIndex, 0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		m_g[offsetSup + m_dimSup] -= delta*ui.getScaling(cIndex, 0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			m_g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			m_g[offsetSup + s] -= delta*ui.getScaling(cIndex, k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			m_g[offsetSup + m_dimSup + s] -= delta*ui.getScaling(cIndex, k)*fv.getValue(); // a_i*x_di
		}
	}
	protected double calculateR1(){
		double R1 = 0;
		int offset;
		for(int i=0; i<m_k; i++){
			offset = i*m_dim*2;
			for(int k=0; k<m_dim; k++){
				R1 += m_eta1 * (m_A[offset+k]-1) * (m_A[offset+k]-1);//(a[i]-1)^2
				R1 += m_eta2 * m_A[offset+m_dim+k] *m_A[offset+m_dim+k];//b[i]^2
			}
		}
		return R1;
	}
	
	// Gradients for the gs.
	protected void gradientByR1(){
		int offset;
		for(int k=0;k<m_k; k++){
			offset = m_dim*2*k;
			for(int i=0; i < m_dim; i++){
				m_g[offset + i] += 2 * m_eta1 * (m_A[offset + i] - 1);
				m_g[offset + i + m_dim] += 2 * m_eta2 * m_A[offset + i + m_dimSup];
			}
		}
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		int offset;
		
		// Init m_sWeights with global weights;
		m_sWeights = new double[m_featureSize + 1];
		System.arraycopy(m_gWeights, 0, m_sWeights, 0, m_gWeights.length);
		
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		_CoLinAdaptStruct.sharedA = new double[getVSize()];
		m_A = _CoLinAdaptStruct.sharedA;

		// Init shared A matrix.
		for(int k=0; k<=m_k; k++) {
			offset = 2*m_dim*k;
			for(int d=0; d<m_dim; d++)
				m_A[offset+d] = 1;
			if(k==m_k){
				for(int d=0; d<m_dimSup; d++)
				m_A[offset+d] = 1;
			}
		}
	}
	
	@Override
	public int getVSize(){
		return m_dim*2*m_k + m_dimSup*2;
	}
	
	// We can do A_i*A_s*w_g*x at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int n = 0, k = 0, cIndex = 0; // feature index and feature group index
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
		cIndex = m_userClusterIndex[ui.getId()];
		double value = ui.getScaling(cIndex, 0)*getSupWeights(0) + ui.getShifting(cIndex, 0);//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (ui.getScaling(cIndex, k)*getSupWeights(n) + ui.getShifting(cIndex, k)) * fv.getValue();
		}
		return Utils.logistic(value);
	}
	@Override
	// In the algorithm, each individual user's model is A_i*A_s*w_g.
	protected void setPersonalizedModel() {
		int gid, cIndex;
		_CoLinAdaptStruct ui;
		
		//get the model weight for super user
		for(int n=0; n<=m_featureSize; n++)
			m_sWeights[n] = getSupWeights(n);
		
		//Update each user's personalized model.
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)m_userList.get(i);
			cIndex = m_userClusterIndex[ui.getId()];
			//set the other features
			for(int n=0; n<=m_featureSize; n++) {
				gid = m_featureGroupMap[n];
				m_pWeights[n] = ui.getScaling(cIndex, gid) * m_sWeights[n] + ui.getShifting(cIndex, gid);
			}
			ui.setPersonalizedModel(m_pWeights);
		}
	}
	
	// w_s = A_s * w_g
	public double getSupWeights(int index){
		int gid = m_featureGroupMap4SupUsr[index], offsetSup = m_k* 2 * m_dim;
		return m_A[offsetSup + gid] * m_gWeights[index] + m_A[offsetSup + gid + m_dimSup];
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, offsetSup;
		double magA = 0, magB = 0;
		for(int n=0; n<m_k; n++) {
			offset = n*vSize;
			for(int i=0; i<m_dim; i++){
				magA += m_g[offset+i]*m_g[offset+i];
				magB += m_g[offset+m_dim+i]*m_g[offset+m_dim+i];
			}
		}

		double magASup = 0, magBSup = 0;
		offsetSup = vSize * m_k;
		for(int i=0; i<m_dimSup; i++){
			magASup += m_g[offsetSup+i] * m_g[offsetSup+i];
			magBSup += m_g[offsetSup+m_dimSup+i] * m_g[offsetSup + m_dimSup+i];
		}
		
		if (m_displayLv==2)
			System.out.format("\tuser(%.4f,%.4f), super user(%.4f,%.4f)\n", magA, magB, magASup, magBSup);
		return magA + magB;
	}
	public int getIndividualOffset(_CoLinAdaptStruct ui){
		return m_userClusterIndex[ui.getId()]*m_dim*2;
	}
	public int getSupOffset(){
		return m_k*m_dim*2;
	}	
	
	@Override
	public String toString() {
		return String.format("MTLinAdaptWithClusters[dim:%d, supDim:%d, eta1:%.3f,eta2:%.3f,lambda1:%.3f,lambda2:%.3f, personalized:%b]", 
				m_dim, m_dimSup, m_eta1, m_eta2, m_eta3, m_eta4, m_personalized);
	}
	
	
	@Override
	public double train() {
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue, oldFValue = Double.MAX_VALUE;
		int vSize = getVSize(), displayCount = 0;
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
					gradientByFunc(user);
				}
				// The contribution from R^1(A_s) to both function value and gradients.
				fValue += calculateR1();
				fValue += calculateRs(); // + R^1(A_s)
				gradientByR1();
				gradientByRs(); // Gradient from R^1(A_s)

				if (m_displayLv == 2) {
					System.out.format("Fvalue is %.3f", fValue);

					gradientTest();
				} else if (m_displayLv == 1) {
					if (fValue < oldFValue)
						System.out.print("o");
					else
						System.out.print("x");

					if (++displayCount % 100 == 0)
						System.out.println();
				}
				oldFValue = fValue;
//				LBFGS.lbfgs(vSize, 6, m_A, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);// In the training process, A is updated.

				LBFGS.lbfgs(vSize, 5, m_A, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);// In the training process, A is updated.
			} while (iflag[0] != 0);
			System.out.println();
		} catch (ExceptionWithIflag e) {
			System.err.println("********lbfgs fails here!******");
			e.printStackTrace();
			m_lbfgs = 0;
		}

		setPersonalizedModel();
		return oldFValue;
	}
	
}
