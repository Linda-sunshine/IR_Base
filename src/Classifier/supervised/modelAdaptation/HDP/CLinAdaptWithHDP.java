package Classifier.supervised.modelAdaptation.HDP;

import java.util.ArrayList;
import java.util.HashMap;
import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.DirichletProcess.DoubleNormalPrior;
/***
 * In this class, we utilize HDP to do automatic grouping based on each review.
 * The method does not involve global model learning.
 * @author lin
 */
public class CLinAdaptWithHDP extends CLRWithHDP {

	protected double[] m_abNuB = new double[]{1, 0.1}; // prior for scaling
	public static double[] m_supWeights; // newly learned global model, dummy variable in CLinAdaptWithDP
	
	public CLinAdaptWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, lm);
		loadFeatureGroupMap(featureGroupMap);
		_HDPAdaptStruct.m_featureGroupMap = m_featureGroupMap;//this is really an ugly solution
		m_supWeights = m_gWeights;// this design is for evaluate purpose since we don't need to rewrite evaluate.
	}
	
	public CLinAdaptWithHDP(int classNo, int featureSize, String globalModel, String featureGroupMap, double[] lm) {
		super(classNo, featureSize, globalModel, lm);
		loadFeatureGroupMap(featureGroupMap);
		_HDPAdaptStruct.m_featureGroupMap = m_featureGroupMap;//this is really an ugly solution
		m_supWeights = m_gWeights;// this design is for evaluate purpose since we don't need to rewrite evaluate.
	}

	public CLinAdaptWithHDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel, double[] lm) {
		super(classNo, featureSize, featureMap, globalModel, lm);
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2;
	}
	
	@Override
	protected void initPriorG0() {
		m_G0 = new DoubleNormalPrior(m_abNuB[0], m_abNuB[1], m_abNuA[0], m_abNuA[1]);
	}
	
	@Override
	protected boolean isTransformationBased() {
		return true;
	}
	
	@Override
	// R1 over each cluster, R1 over super cluster.
	protected double calculateR1(){
		double R1 = 0;
		int offset;
		
		// scan through all clusters
		for(int i=0; i<m_kBar; i++) {
			//likelihood over prior
			R1 += m_G0.logLikelihood(m_hdpThetaStars[i].getModel(), m_eta1, m_eta2);
		
			//gradient over prior
			offset = m_dim*2*i;
			for(int k=0; k<m_dim;k++){
				m_g[offset+k] += m_eta1 * (m_models[offset+k]-m_abNuB[0])/m_abNuB[1]/m_abNuB[1]; //scaling
				m_g[offset+k+m_dim] += m_eta2 * (m_models[offset+k+m_dim]-m_abNuA[0])/m_abNuA[1]/m_abNuA[1]; // shifting
			}
		}
		return R1;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_Review r = (_Review) review;

		int n, k; // feature index
		int cIndex = r.getHDPThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error, cannot find the theta star!");
		int offset = m_dim*2*cIndex;
		
		double delta = (review.getYLabel() - logit(review.getSparse(), r)) * weight;
		
		// Bias term for individual user.
		g[offset] -= delta*m_gWeights[0]; //a[0] = ws0*x0; x0=1
		g[offset + m_dim] -= delta;//b[0]
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			g[offset + k] -= delta*m_gWeights[n]*fv.getValue(); // w_si*x_di
			g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
		}
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		// Init each user.
		for(_User user:userList)
			m_userList.add(new _HDPAdaptStruct(user, m_dim));

		m_pWeights = new double[m_gWeights.length];		
	}
	
	@Override
	protected double logit(_SparseFeature[] fvs, _Review r){
		int k, n;
		double[] Au = r.getHDPThetaStar().getModel(); 
		double sum = Au[0]*m_gWeights[0] + Au[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			sum += (Au[k]*m_gWeights[n] + Au[m_dim+k]) * fv.getValue();
		}
		return Utils.logistic(sum);
	}
	
	// Assign the optimized models to the clusters.
	@Override
	protected void setThetaStars(){
		// Assign models to clusters.
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_models, m_dim*2*i, m_hdpThetaStars[i].getModel(), 0, m_dim*2);
	}
	
	@Override
	public String toString() {
		return String.format("CLinAdaptWithHDP[dim:%d,lmDim:%d,M:%d,alpha:%.4f,eta:%.4f,beta:%.4f,nScale:(%.3f,%.3f),#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]",m_dim,m_lmDim,m_M,m_alpha,m_eta,m_beta,m_eta1,m_eta2,m_numberOfIterations,m_abNuA[0],m_abNuA[1],m_abNuB[0],m_abNuB[1]);
	}

	public void setsdB(double s){
		m_abNuB[1] = s;
	}
}
