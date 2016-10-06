package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._SparseFeature;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class LinAdaptOverall extends LinAdapt {
	double[] m_sharedA; // The transformation matrix shared by all the users/super user.
	double[] m_glbWeights; // Store the learned the global weights shared by all users.
	
	public LinAdaptOverall(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public void initLBFGS(){
		if(m_g == null)
			m_g = new double[super.getVSize()];
		if(m_diag == null)
			m_diag = new double[super.getVSize()];
		// Init m_A
		m_sharedA = new double[super.getVSize()];
		for(int k=0; k<m_dim; k++){
			m_sharedA[k] = 1;
		}
	}
	
	//Calculate the function value of the new added instance.
	@Override
	protected double calculateFuncValue(_AdaptStruct u){
		return  calcLogLikelihood(u); //log likelihood.
	}
	
	// We can do A*w*x at the same time to reduce computation.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		double value = m_sharedA[0]*m_gWeights[0] + m_sharedA[m_dim];//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (m_sharedA[k]*m_gWeights[n] + m_sharedA[k + m_dim]) * fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (m_sharedA[i]-1) * (m_sharedA[i]-1);//(a[i]-1)^2
			R1 += m_eta2 * m_sharedA[i + m_dim] * m_sharedA[i + m_dim];//b[i]^2
		}
		return R1;
	}
	
	//Calculate the gradients for the use in LBFGS.
	@Override
	protected void calculateGradients(_AdaptStruct user){
		gradientByFunc(user);
	}
	
	//shared gradient calculation by batch and online updating
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		
		int n, k; // feature index and feature group index		
		double delta = (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[0] -= weight*delta*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[m_dim] -= weight*delta;//b[0]

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[k] -= weight * delta * m_gWeights[n] * fv.getValue();
			m_g[m_dim + k] -= weight * delta * fv.getValue();  
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(){
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[k] += 2 * m_eta1 * (m_sharedA[k]-1);// add 2*eta1*(a_k-1)
			m_g[k + m_dim] += 2 * m_eta2 * m_sharedA[m_dim + k]; // add 2*eta2*b_k
		}
	}
	
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue = 0, L, displayCount = 0, oldFValue = Double.MAX_VALUE;
		
		init();
		initLBFGS();
		try{	
			do{
				L = 0;
				Arrays.fill(m_g, 0); // initialize gradient					
				for(_AdaptStruct user:m_userList) {			
					L += calculateFuncValue(user);
					calculateGradients(user);
				}
				// The shared R1 regularization.
				fValue = calculateR1() - L;
				gradientByR1();
				
				if (m_displayLv==2) {
					System.out.println("Fvalue is " + fValue);
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					if (++displayCount%100==0)
						System.out.println();
				} 
				oldFValue = fValue;
					
				LBFGS.lbfgs(m_sharedA.length, 6, m_sharedA, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				calculateGlobalWeights();
				setPersonalizedModel();
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
		return oldFValue;
	}
	
	public void calculateGlobalWeights(){
		int gid;
		m_glbWeights = new double[m_featureSize + 1];
		
		//set bias term
		m_glbWeights[0] = m_sharedA[0] * m_gWeights[0] + m_sharedA[m_dim];
		
		//set the other features
		for(int n=0; n<m_featureSize; n++) {
			gid = m_featureGroupMap[1+n];
			m_glbWeights[1+n] = m_sharedA[gid] * m_gWeights[1+n] + m_sharedA[gid + m_dim];
		}	
	}
	
	@Override
	protected void setPersonalizedModel() {
		for(_AdaptStruct u: m_userList){
			u.setPersonalizedModel(m_glbWeights);
		}
	}
}
