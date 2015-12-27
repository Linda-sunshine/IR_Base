package Classifier.semisupervised.CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import Classifier.BaseClassifier;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;

public class LinAdapt extends BaseClassifier {
	ArrayList<_LinAdaptStruct> m_userList;//a list of users for adapting personalized models
	
	//The number of feature groups k, so the total number of dimensions of weights is 2k+2.	
	int m_dim;//The dimension of scaling and shifting parameters (including the bias term)
	int[] m_featureGroupMap; // bias term is at position 0
	double[] m_gWeights; //global model weight
	double[] m_pWeights; // cache for personalized weight
	
	//Trade-off parameters	
	double m_eta1; // weight for scaling.
	double m_eta2; // weight for shifting.
	
	//shared space for LBFGS optimization
	double[] m_diag; //parameter used in lbfgs.
	double[] m_g;//optimized gradients. 
	
	public LinAdapt(int classNo, int featureSize, String globalModel, String featureGroupMap){
		super(classNo, featureSize);
		m_userList = null;				
		m_pWeights = null;
		
		loadFeatureGroupMap(featureGroupMap);
		loadGlobalModel(globalModel);
		
		// default value of trade-off parameters
		m_eta1 = 0.5;
		m_eta2 = 0.5;
	}  
	
	public void setTradeOffs(double eta1, double eta2) {
		m_eta1 = eta1;
		m_eta2 = eta2;
	}
	
	/***When we do feature selection, we will group features and store them in file. 
	 * The index is the index of features and the corresponding number is the group index number.***/
	public void loadFeatureGroupMap(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");//Group information of each feature.
			reader.close();
			
			m_featureGroupMap = new int[features.length + 1]; //One more term for bias, bias->0.
			m_dim = 0;
			//Group index starts from 0, so add 1 for it.
			for(int i=0; i<features.length; i++) {
				m_featureGroupMap[i+1] = Integer.valueOf(features[i]) + 1;
				if (m_dim < m_featureGroupMap[i+1])
					m_dim = m_featureGroupMap[i+1];
			}
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	//Load global model from file.
	public void loadGlobalModel(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String[] features = reader.readLine().split(",");
			reader.close();
			
			m_gWeights = new double[features.length];
			for(int i=0; i<features.length; i++)
				m_gWeights[i] = Double.valueOf(features[i]);
			
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	//Initialize the weights of the transformation matrix.
	public void loadUsers(ArrayList<_User> userList){	
		m_userList = new ArrayList<_LinAdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _LinAdaptStruct(user, m_dim));
		m_pWeights = new double[m_gWeights.length];
	}
	
	protected void initLBFGS(){
		if(m_g == null)
			m_g = new double[m_dim*2];
		if(m_diag == null)
			m_diag = new double[m_dim*2];
		
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_g, 0);
	}

	// We can do A*w*x at the same time to reduce computation.
	protected double logit(_SparseFeature[] fvs, _LinAdaptStruct user){
		double value = user.getScaling(0)*m_gWeights[0] + user.getShifting(0);//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (user.getScaling(k)*m_gWeights[n] + user.getShifting(k)) * fv.getValue();
		}
		return 1/(1+Math.exp(-value));
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_LinAdaptStruct user){
		double L = 0; //log likelihood.
		double Pi = 0, R1 = 0;
		
		for(_Review review: user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			Pi = logit(review.getSparse(), user);
			if(review.getYLabel() == 1)
				L += Math.log(Pi);
			else 
				L += Math.log(1 - Pi);
		}
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (user.getScaling(i)-1) * (user.getScaling(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * user.getShifting(i) * user.getShifting(i);//b[i]^2
		}
		
		return R1 - L;
	}
	
	protected void gradientByFunc(_LinAdaptStruct user) {
		double delta;
		int n, k; // feature index and feature group index
		
		//Update gradients one review by one review.
		for(_Review review: user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			delta = review.getYLabel() - logit(review.getSparse(), user);
			
			//Bias term.
			m_g[0] -= delta*m_gWeights[0]; //a[0] = w0*x0; x0=1
			m_g[m_dim] -= delta;//b[0]
			
			//Traverse all the feature dimension to calculate the gradient.
			for(_SparseFeature fv: review.getSparse()){
				n = fv.getIndex() + 1;
				k = m_featureGroupMap[n];
				m_g[k] -= delta * m_gWeights[n] * fv.getValue();
				m_g[m_dim + k] -= delta * fv.getValue();  
			}
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(_LinAdaptStruct user){
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[k] += 2 * m_eta1 * (user.getScaling(k)-1);// add 2*eta1*(a_k-1)
			m_g[k+m_dim] += 2 * m_eta2 * user.getShifting(k); // add 2*eta2*b_k
		}
	}
		
	//Calculate the gradients for the use in LBFGS.
	protected void calculateGradients(_LinAdaptStruct user){
		gradientByFunc(user);
		gradientByR1(user);
	}
	
	protected double gradientTest() {
		double magA = 0, magB = 0 ;
		for(int i=0; i<m_dim; i++){
			magA += m_g[i]*m_g[i];
			magB += m_g[i+m_dim]*m_g[i+m_dim];
		}
		System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is batch training in each individual user
	public void train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, A[];
		int vSize = 2*m_dim;
		
		for(_LinAdaptStruct user:m_userList) {
			initLBFGS();
			try{
				A = user.getA();
				do{
					fValue = calculateFuncValue(user);
					System.out.println("Fvalue is " + fValue);
					
					Arrays.fill(m_g, 0); // initialize gradient
					calculateGradients(user);
					gradientTest();
					
					LBFGS.lbfgs(vSize, 5, A, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-10, iflag);//In the training process, A is updated.
				} while(iflag[0] != 0);
			} catch(ExceptionWithIflag e) {
				e.printStackTrace();
			}
			setPersonalizedModel(user);
		}
	}
	
	void setPersonalizedModel(_LinAdaptStruct user) {
		int gid;
		
		//set bias term
		m_pWeights[0] = user.getScaling(0) * m_gWeights[0] + user.getShifting(0);
		
		//set the other features
		for(int i=0; i<m_featureSize; i++) {
			gid = m_featureGroupMap[1+i];
			m_pWeights[1+i] = user.getScaling(gid) * m_gWeights[1+i] + user.getShifting(gid);
		}
		user.setPersonalizedModel(m_pWeights, m_classNo, m_featureSize);
	}
	
	//Batch mode: given a set of reviews and accumulate the TP table.
	@Override
	public double test(){
		int trueL = 0, predL = 0;
		int[][] TPTable = new int[2][2];
		for(_LinAdaptStruct user:m_userList) {
			for(_Review r:user.getReviews()) {
				if (r.getType() != rType.TEST)
					continue;
				trueL = r.getYLabel();
				predL = user.predict(r); // evoke user's own model
				TPTable[trueL][predL]++; // macro average
			}
		}
		
		return 0;
	}

	@Override
	public void train(Collection<_Doc> trainSet) {	
		System.err.println("[Error]train(Collection<_Doc> trainSet) is not implemented in LinAdapt!");
		System.exit(-1);
	}

	@Override
	public int predict(_Doc doc) {
		System.err.println("[Error]predict(_Doc doc) is not implemented in LinAdapt!");
		System.exit(-1);
		return 0;
	}

	@Override
	public double score(_Doc d, int label) {
		System.err.println("[Error]score(_Doc d, int label) is not implemented in LinAdapt!");
		System.exit(-1);
		return 0;
	}

	@Override
	protected void init() {		
		System.err.println("[Error]init() is not implemented in LinAdapt!");
		System.exit(-1);
	}

	@Override
	protected void debug(_Doc d) {
		System.err.println("[Error]debug(_Doc d) is not implemented in LinAdapt!");
		System.exit(-1);
	}

	@Override
	public void saveModel(String modelLocation) {
		System.err.println("[Error]saveModel(String modelLocation) is not implemented in LinAdapt!");
		System.exit(-1);
	}
}
