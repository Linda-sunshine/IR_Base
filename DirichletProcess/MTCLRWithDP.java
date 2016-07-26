package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.HashMap;

import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
/**
 * In the class, we extend the CLR to multi-task learning.
 * Intead of clusters, we also have a global part.
 * @author lin
 *
 */
public class MTCLRWithDP extends CLRWithDP {
	protected double m_q;// the wc + m_q*wg;
	public static double[] m_supWeights; // newly learned global model

	public MTCLRWithDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize, featureMap, globalModel);
		m_supWeights = new double[m_dim];
		m_q = 0.001;
	}
	@Override
	protected void accumulateClusterModels(){
		m_models = new double[getVSize()];
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_thetaStars[i].getModel(), 0, m_models, m_dim*i, m_dim);
		System.arraycopy(m_supWeights, 0, m_models, m_dim*m_kBar, m_dim);
	}
	
	@Override
	protected int getVSize(){
		return m_dim*(m_kBar+1);
	}
	
	protected void initPriorG0(){
		super.initPriorG0();
		m_G0.sampling(m_supWeights);// sample super user's weights.
	}

	@Override	
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		double[] weights = addTwoArrays(((_DPAdaptStruct)u).getThetaStar().getModel(), m_supWeights, m_q);
		double sum = Utils.dotProduct(weights, fvs, 0);
		return Utils.logistic(sum);
	}
	protected double[] addTwoArrays(double[] a1, double[] a2, double q){
		if(a1.length != a2.length)
			return null;
		double[] arr = new double[a1.length];
		for(int i=0; i<a1.length; i++){
			arr[i] = a1[i] + q*a2[i];
		}
		return arr;
	}
	@Override
	protected double calculateR1(){
		double R1 = super.calculateR1();
		R1 += m_G0.logLikelihood(m_supWeights, m_eta2, 0);
		return R1;
		
	}
	
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;		
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*cIndex, offsetSup = m_dim*m_kBar;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[offset] -= delta; //x0=1, each cluster.
		m_g[offsetSup] -= m_q*delta; // super model.

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= delta * fv.getValue();// cluster model.
			m_g[offsetSup + n] -= delta * fv.getValue() * m_q;// super model.
		}
	}
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;		
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*cIndex, offsetSup = m_dim*m_kBar;
		double delta = weight * (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		g[offset] -= delta; //x0=1, each cluster.
		g[offsetSup] -= m_q*delta; // super model.

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			g[offset + n] -= delta * fv.getValue();// cluster model.
			g[offsetSup + n] -= delta * fv.getValue() * m_q;// super model.
		}
	}
	
	@Override
	protected void setPersonalizedModel() {
		_DPAdaptStruct user;
		double[] pWeights;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			pWeights = addTwoArrays(user.getThetaStar().getModel(), m_supWeights, m_q);
			user.setPersonalizedModel(pWeights);
		}
	}
	
	protected void setThetaStars(){
		super.setThetaStars();
		System.arraycopy(m_models, m_kBar*m_dim, m_supWeights, 0, m_dim);
	}
	// Gradient by the regularization.
	protected void gradientByR1(){
		// cluster part.
		for(int i=0; i<m_g.length-m_dim; i++)
			m_g[i] += m_eta1 * (m_models[i]-m_gWeights[i%m_dim])/(m_abNuA[1]*m_abNuA[1]);
		// super model part.
		for(int i=m_kBar*m_dim; i<m_g.length; i++)
			m_g[i] += m_eta2 * (m_models[i]-m_gWeights[i%m_dim])/(m_abNuA[1]*m_abNuA[1]);
	}
	@Override
	public String toString() {
		return String.format("MTCLRWithDP[dim:%d,q:%.4f,M:%d,alpha:%.4f,nScale:%.3f,#Iter:%d,N(%.3f,%.3f)]", m_dim,m_q, m_M, m_alpha, m_eta1, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
		
}
