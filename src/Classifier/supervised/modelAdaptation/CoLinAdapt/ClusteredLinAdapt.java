package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._SparseFeature;
import structures._User;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

/***
 * In this class, we would like to incorporate the cluster information 
 * to see how it influences the final performance. 
 * @author lin
 */
public class ClusteredLinAdapt extends LinAdapt{
	int m_clusterSize; // The cluster number.
	// Parameters for different parts.
	double m_u = 1; // global parts.
	double m_c = 1; // cluster parts.
	double m_i = 1; // individual parts.
	int[] m_userClusterIndex; // The index is user index, the value is corresponding cluster no.

	public ClusteredLinAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, int clusterNo, int[] userClusterIndex) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_clusterSize = clusterNo;
		m_userClusterIndex = userClusterIndex;
	}
	
	public void setParameters(double u, double c, double i){
		m_u = u;
		m_c = c;
		m_i = i;
	}
	
	@Override
	public String toString() {
		return String.format("ClusteredLinAdapt[dim:%d,kmeans:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_clusterSize, m_eta1, m_eta2);
	}
	
	//Initialize the weights of the transformation matrix.
	@Override
	public void loadUsers(ArrayList<_User> userList){
		int totalUserSize = userList.size();
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _ClusterLinAdaptStruct(user, m_dim, i, totalUserSize, m_clusterSize));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		// step1: init the shared A: individual + cluster + global
		_ClusterLinAdaptStruct.sharedA = new double[getVSize()];
		for(int i=0; i<m_userList.size()+m_clusterSize+1; i++){
			for(int j=0; j<m_dim; j++){
				_ClusterLinAdaptStruct.sharedA[i*m_dim*2+j] = 1;
			}
		}
	}

	@Override
	int getVSize() {
		return m_dim*2*(m_userList.size() + m_clusterSize + 1);
	}
	
	protected double linearFunc(_SparseFeature[] fvs, _AdaptStruct u) {
		
		_ClusterLinAdaptStruct user = (_ClusterLinAdaptStruct)u;
		int clusterIndex = m_userClusterIndex[user.getId()];
		double scaling, shifting;
		scaling = m_u*user.getGlobalScaling(0) + m_c*user.getClusterScaling(clusterIndex, 0) + m_i*user.getScaling(0);
		shifting = m_u*user.getGlobalShifting(0) + m_c*user.getClusterShifting(clusterIndex, 0) + m_i*user.getShifting(0);
		double value = scaling*m_gWeights[0] + shifting;//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			scaling = m_u*user.getGlobalScaling(k) + m_c*user.getClusterScaling(clusterIndex, k) + m_i*user.getScaling(k);
			shifting = m_u*user.getGlobalShifting(k) + m_c*user.getClusterShifting(clusterIndex, k) + m_i*user.getShifting(k);
			value += (scaling*m_gWeights[n] + shifting) * fv.getValue();
		}
		return value;
	}	
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		double fValue = super.calculateFuncValue(u);
		double Rs = calculateRs();
		return fValue + Rs;
	}
	
	// Shall we regularize the Rc or Rs?
	protected double calculateRs(){
		return 0;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_ClusterLinAdaptStruct user = (_ClusterLinAdaptStruct)u;

		int n, k; // feature index and feature group index		
		int clusterIndex = m_userClusterIndex[user.getId()];
		int iOffset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
		int cOffset = 2*m_dim*(m_userList.size() + clusterIndex);
		int gOffset = 2*m_dim*(m_userList.size() + m_clusterSize);
		double delta = (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term for individual part.
		m_g[iOffset] -= weight*delta*m_i*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[iOffset + m_dim] -= weight*delta*m_i;//b[0]

		//Bias term for cluster part.
		m_g[cOffset] -= weight*delta*m_c*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[cOffset + m_dim] -= weight*delta*m_c;//b[0]
		
		//Bias term for global part.
		m_g[gOffset] -= weight*delta*m_gWeights[0]*m_u; //a[0] = w0*x0; x0=1
		m_g[gOffset + m_dim] -= weight*delta*m_u;//b[0]
				
		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			
			//Individual part.
			m_g[iOffset + k] -= weight*delta*m_i*m_gWeights[n]*fv.getValue();
			m_g[iOffset + m_dim + k] -= weight*delta*m_i*fv.getValue(); 
			
			//Cluster part.
			m_g[cOffset + k] -= weight*delta*m_c*m_gWeights[n]*fv.getValue();
			m_g[cOffset + m_dim + k] -= weight*delta*m_c*fv.getValue(); 
			
			//Global part.
			m_g[gOffset + k] -= weight*delta*m_u*m_gWeights[n]*fv.getValue();
			m_g[gOffset + m_dim + k] -= weight*delta*m_u*fv.getValue(); 
		}
	}
	
	protected void initPerIter() {
		Arrays.fill(m_g, 0); // initialize gradient
	}
	
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, oldFValue = Double.MAX_VALUE, totalFvalue = 0;
		int displayCount = 0;
		_LinAdaptStruct user;

		init();
		initLBFGS();

		try{
			do{
				fValue = 0;
				initPerIter();
					
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_LinAdaptStruct)m_userList.get(i);
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
					
				if (m_displayLv==2) {
					gradientTest();
					System.out.println("Fvalue is " + fValue);
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					 
					if (++displayCount%100==0)
					System.out.println();
				}
				LBFGS.lbfgs(m_g.length, 6, _ClusterLinAdaptStruct.sharedA, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-32, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}
		setPersonalizedModel();
		return totalFvalue;
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, iOffset, cOffset;
		double magA = 0, magB = 0, magC = 0, magD = 0;
		// gradients for the individual parts.
		for(int n=0; n<m_userList.size(); n++) {
			for(int i=0; i<m_dim; i++){
				iOffset = n*vSize + i;
				magA += m_g[iOffset]*m_g[iOffset];
				magB += m_g[iOffset+m_dim]*m_g[iOffset+m_dim];
			}
		}
		// gradients for the cluster part.
		for(int c=0; c<m_clusterSize; c++){
			for(int i=0; i<m_dim; i++){
				cOffset = (c + m_userList.size())*vSize + i;
				magC += m_g[cOffset]*m_g[cOffset];
				magD += m_g[cOffset+m_dim]*m_g[cOffset+m_dim];
			}
		}
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a:%.5f,b:%.5f,c:%.5f,d:%.5f\n", magA, magB, magC, magD);
		return magA + magB;
	}
	
	@Override
	protected void setPersonalizedModel() {
		int gid;
		_ClusterLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_ClusterLinAdaptStruct)m_userList.get(i);
			
			int clusterIndex = m_userClusterIndex[user.getId()];
			double scaling, shifting;
			// Bias term.
			scaling = m_u*user.getGlobalScaling(0) + m_c*user.getClusterScaling(clusterIndex, 0) + m_i*user.getScaling(0);
			shifting = m_u*user.getGlobalShifting(0) + m_c*user.getClusterShifting(clusterIndex, 0) + m_i*user.getShifting(0);
			m_pWeights[0] = scaling*m_gWeights[0] + shifting;//Bias term: w0*a0+b0.
			
			for(int n=0; n<m_featureSize; n++){
				gid = m_featureGroupMap[1+n];
				scaling = m_u*user.getGlobalScaling(gid) + m_c*user.getClusterScaling(clusterIndex, gid) + m_i*user.getScaling(gid);
				shifting = m_u*user.getGlobalShifting(gid) + m_c*user.getClusterShifting(clusterIndex, gid) + m_i*user.getShifting(gid);
				m_pWeights[1+n] = scaling* m_gWeights[1+n] + shifting;
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
}
