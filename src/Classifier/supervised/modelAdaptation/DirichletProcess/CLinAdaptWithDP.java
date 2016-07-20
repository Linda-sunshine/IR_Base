package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import utils.Utils;
/***
 * Linear transformation matrix with DP.
 * @author lin
 */
public class CLinAdaptWithDP extends CLogisticRegressionWithDP{
	protected int m_dimSup;
	protected int[] m_featureGroupMap4SupUsr; // bias term is at position 0
	protected double[] m_supModel; // linear transformation.
	protected double[] m_sWeights; // weights of super user.
	protected double m_eta3, m_eta4;// coefficients in front of super model.
	
	public CLinAdaptWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap, String featureGroup4Sup){
		super(classNo, featureSize, featureMap, globalModel);
		loadFeatureGroupMap(featureGroupMap);
		loadFeatureGroupMap4SupUsr(featureGroup4Sup);
		m_supModel = new double[m_dimSup*2]; // globally shared transformation matrix.
		m_eta3 = 0.01;
		m_eta4 = 0.01;
	}
	
	@Override
	protected void accumulateClusterModels(){
		m_models = new double[getVSize()];
		for(int i=0; i<m_kBar; i++){
			System.arraycopy(m_thetaStars[i].m_beta, 0, m_models, m_dim*2*i, m_dim*2);
		}
		// we put the global part in the end.
		System.arraycopy(m_supModel, 0, m_models, m_dim*2*m_kBar, m_dim*2);
	}
	
	@Override
	// R1 over each cluster, R1 over super cluster.
	protected double calculateR1(){
		double R1 = 0;
		int offset;
		// Clusters.
		for(int i=0; i<m_kBar; i++){
			offset = m_dim*2*i;
			for(int j=0; j<m_dim; j++){
				R1 += m_eta1*(m_models[offset+j]-1)*(m_models[offset+j]-1);
				R1 += m_eta2*m_models[offset+m_dim+j]*m_models[offset+m_dim+j];
			}
		}
		// Super model.
		for(int i=0; i<m_dimSup; i++){
			R1 += m_eta3*(m_supModel[i]-1)*(m_supModel[i]-1);
			R1 += m_eta4*m_supModel[i+m_dimSup]*m_supModel[i+m_dimSup];
		}
		return R1;
	}
	
	protected double getSupWeights(int n){
		int gid = m_featureGroupMap4SupUsr[n];
		return m_supModel[gid]*m_gWeights[n] + m_supModel[gid+m_dimSup];
		
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2 + m_dimSup*2;// we have global here.
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n, k, s; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		int offset = m_dim*2*cIndex, offsetSup = m_dim*2*m_kBar;
		double delta = (review.getYLabel() - logit(review.getSparse(), user, cIndex));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		// Bias term for individual user.
		m_g[offset] -= delta*getSupWeights(0); //a[0] = ws0*x0; x0=1
		m_g[offset + m_dim] -= delta;//b[0]

		// Bias term for super user.
		m_g[offsetSup] -= delta*user.getScaling(0)*m_gWeights[0]; //a_s[0] = a_i0*w_g0*x_d0
		m_g[offsetSup + m_dimSup] -= delta*user.getScaling(0); //b_s[0] = a_i0*x_d0
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= delta*getSupWeights(n)*fv.getValue(); // w_si*x_di
			m_g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
			
			s = m_featureGroupMap4SupUsr[n];
			m_g[offsetSup + s] -= delta*user.getScaling(k)*m_gWeights[n]*fv.getValue(); // a_i*w_gi*x_di
			m_g[offsetSup + m_dimSup + s] -= delta*user.getScaling(k)*fv.getValue(); // a_i*x_di
		}
	}
	
	// Gradient by the regularization.
	@Override
	protected void gradientByR1(){
		int offset;
		for(int i=0; i<m_kBar; i++){
			offset = m_dim*2*i;
			for(int k=0; k<m_dim;k++){
				m_g[offset+k] += 2*m_eta1*(m_models[offset+k]-1);
				m_g[offset+k+m_dim] += 2*m_eta2*(m_models[offset+k+m_dim]);
			}
		}
		// R1 by super model.
		offset = m_dim*2*m_kBar;
		for(int k=0; k<m_dimSup; k++){
			m_g[offset+k] += 2*m_eta3*(m_supModel[k]-1);
			m_g[offset+k+m_dimSup] += 2*m_eta4*m_supModel[m_dimSup+k];
		}
	}
	
	@Override
	protected double gradientTest() {
		double magC = 0, magS = 0 ;
		int offset = m_dim*2*m_kBar;
		for(int i=0; i<offset; i++)
			magC += m_g[i]*m_g[i];
		for(int i=offset; i<m_g.length; i++)
			magS += m_g[i]*m_g[i];
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for clusters: %.5f, super model: %.5f\n", magC, magS);
		return 0;
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		// Init each user.
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user, m_dim));// Difference
		// Init super user model.
		for(int i=0; i<m_dimSup; i++)
			m_supModel[i] = 1;
		m_pWeights = new double[m_gWeights.length];		
	}
	
	// Feature group map for the super user.
	protected void loadFeatureGroupMap4SupUsr(String filename){
		// If there is no feature group for the super user.
		if(filename == null){
			m_dimSup = m_featureSize + 1;
			m_featureGroupMap4SupUsr = new int[m_featureSize + 1]; //One more term for bias, bias->0.
			for(int i=0; i<=m_featureSize; i++)
				m_featureGroupMap4SupUsr[i] = i;
			return;
		} else{// If there is feature grouping for the super user, load it.
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
			} catch(IOException e){
				System.err.format("[Error]Fail to open super user group file %s.\n", filename);
			}
		}
		System.out.format("[Info]Feature group size for super user %d\n", m_dimSup);
	}
	
	// Logit function is different from the father class.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u, int c){
		int k, n;
		double[] As = m_thetaStars[c].m_beta;
		double value = As[0]*getSupWeights(0) + As[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (As[k]*getSupWeights(n) + As[m_dim+k]) * fv.getValue();
		}
		return Utils.logistic(value);
	}
	
	// Sample thetaStars.
	@Override
	protected void sampleThetaStars(int start, int M){
		for(int m=0; m<M; m++){
			m_thetaStars[start+m] = new _thetaStar(2*m_dim);// sacling+shifting
			m_thetaStars[start+m].setBeta(m_normal);
		}
	}
	
	@Override
	protected void setPersonalizedModel() {
		double[] As;
		int ki, ks;
		_DPAdaptStruct user;

		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			As = user.getThetaStar().m_beta;
			m_pWeights = new double[m_gWeights.length];
			for(int n=0; n<=m_featureSize; n++){
				ki = m_featureGroupMap[n];
				ks = m_featureGroupMap4SupUsr[n];
				m_pWeights[n] = As[ki]*(m_supModel[ks]*m_gWeights[n]+m_supModel[ks+m_dimSup])+As[ki+m_dim];
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
	
	public void setRsTradeOffs(double a, double b){
		m_eta3 = a;
		m_eta4 = b;
	}
	
	// Assign the optimized models to the clusters.
	@Override
	protected void setThetaStars(){
		// Assign models to clusters.
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_models, m_dim*2*i, m_thetaStars[i].m_beta, 0, m_dim*2);
		// Assign model to super user.
		System.arraycopy(m_models, m_dim*2*m_kBar, m_supModel, 0, m_dimSup*2);
	}
	
	@Override
	public String toString() {
		return String.format("CLinAdaptWithDP[dim:%d,M:%d,alpha:%.4f,lambda:%.2f,nuOfIter:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_M, m_alpha, m_lambda, m_numberOfIterations, m_eta1, m_eta2);
	}
}
