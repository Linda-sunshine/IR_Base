package Classifier.semisupervised.CoLinAdapt;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import Classifier.BaseClassifier;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._Doc;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class LinAdapt extends BaseClassifier {
	ArrayList<_LinAdaptStruct> m_userList;//a list of users for adapting personalized models
	
	int m_dim;//The number of feature groups k, so the total number of dimensions of weights is 2(k+1).	
	int[] m_featureGroupMap; // bias term is at position 0
	double[] m_gWeights; //global model weight
	double[] m_pWeights; // cache for personalized weight
	
	//Trade-off parameters	
	double m_eta1; // weight for scaling in R1.
	double m_eta2; // weight for shifting in R2.
	
	//shared space for LBFGS optimization
	double[] m_diag; //parameter used in lbfgs.
	double[] m_g;//optimized gradients. 
	
	int m_displayLv = 1;//0: display nothing during training; 1: display the change of objective function; 2: display everything
	TestMode m_testmode; // test mode of different algorithms 
	
	public LinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize);
		m_userList = null;				
		m_pWeights = null;
		
		loadFeatureGroupMap(featureGroupMap);
		loadGlobalModel(featureMap, globalModel);
		
		// default value of trade-off parameters
		m_eta1 = 0.5;
		m_eta2 = 0.5;
		
		// the only test mode for LinAdapt is batch
		m_testmode = TestMode.TM_batch;
	}  
	
	public void setDisplayLv(int level) {
		m_displayLv = level;
	}
	
	public void setR1TradeOffs(double eta1, double eta2) {
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
			m_dim ++;
			
			System.out.format("[Info]Feature group size %d\n", m_dim);
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	//Load global model from file.
	public void loadGlobalModel(HashMap<String, Integer> featureMap, String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, features[];
			int pos;
			
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			while((line=reader.readLine()) != null) {
				features = line.split(":");
				if (features[0].equals("BIAS"))
					m_gWeights[0] = Double.valueOf(features[1]);
				else if (featureMap.containsKey(features[0])){
					pos = featureMap.get(features[0]);
					if (pos>=0 && pos<m_featureSize)
						m_gWeights[pos+1] = Double.valueOf(features[1]);
					else
						System.err.println("[Warning]Unknown feature " + features[0]);
				} else 
					System.err.println("[Warning]Unknown feature " + features[0]);
			}
			
			reader.close();
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
	
	protected int predict(_Doc review, _LinAdaptStruct user) {
		if (review==null)
			return -1;
		else
			return logit(review.getSparse(), user)>0.5?1:0;
	}
	
	//Calculate the function value of the new added instance.
	protected double calculateFuncValue(_LinAdaptStruct user){
		double L = 0; //log likelihood.
		double Pi = 0, R1 = 0;
		
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			Pi = logit(review.getSparse(), user);
			if(review.getYLabel() == 1) {
				if (Pi>0.0)
					L += Math.log(Pi);					
				else
					L -= Utils.MAX_VALUE;
			} else {
				if (Pi<1.0)
					L += Math.log(1 - Pi);					
				else
					L -= Utils.MAX_VALUE;
			}
		}
		
		//Add regularization parts.
		for(int i=0; i<m_dim; i++){
			R1 += m_eta1 * (user.getScaling(i)-1) * (user.getScaling(i)-1);//(a[i]-1)^2
			R1 += m_eta2 * user.getShifting(i) * user.getShifting(i);//b[i]^2
		}
		
		return R1 - L /getAdaptationSize(user);
	}
	
	protected void gradientByFunc(_LinAdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue;
			
			gradientByFunc(user, review, 1.0);//weight all the instances equally
		}
	}
	
	protected int getAdaptationSize(_LinAdaptStruct user) {
		return user.getAdaptationSize();
	}
	
	//shared gradient calculation by batch and online updating
	protected void gradientByFunc(_LinAdaptStruct user, _Review review, double weight) {
		int n, k; // feature index and feature group index		
		int offset = 2*m_dim*user.m_id;//general enough to accommodate both LinAdapt and CoLinAdapt
		double delta = (review.getYLabel() - logit(review.getSparse(), user)) / getAdaptationSize(user);

		//Bias term.
		m_g[offset] -= weight*delta*m_gWeights[0]; //a[0] = w0*x0; x0=1
		m_g[offset + m_dim] -= weight*delta;//b[0]

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			m_g[offset + k] -= weight * delta * m_gWeights[n] * fv.getValue();
			m_g[offset + m_dim + k] -= weight * delta * fv.getValue();  
		}
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR1(_LinAdaptStruct user){
		int offset = 2*m_dim*user.m_id;//general enough to accommodate both LinAdapt and CoLinAdapt
		//R1 regularization part
		for(int k=0; k<m_dim; k++){
			m_g[offset + k] += 2 * m_eta1 * (user.getScaling(k)-1);// add 2*eta1*(a_k-1)
			m_g[offset + k + m_dim] += 2 * m_eta2 * user.getShifting(k); // add 2*eta2*b_k
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
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is batch training in each individual user
	public void train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, A[], oldFValue = Double.MAX_VALUE;
		int vSize = 2*m_dim;
		
		init();
		for(_LinAdaptStruct user:m_userList) {
			initLBFGS();
			iflag[0] = 0;
			try{
				A = user.getA();
				oldFValue = Double.MAX_VALUE; 
				do{
					Arrays.fill(m_g, 0); // initialize gradient					
					fValue = calculateFuncValue(user);
					calculateGradients(user);
					
					if (m_displayLv==2) {
						System.out.println("Fvalue is " + fValue);
						gradientTest();
					} else if (m_displayLv==1) {
						if (fValue<oldFValue)
							System.out.print("o");
						else
							System.out.print("x");
					} 
					oldFValue = fValue;
					
					LBFGS.lbfgs(vSize, 6, A, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-32, iflag);//In the training process, A is updated.
				} while(iflag[0] != 0);
			} catch(ExceptionWithIflag e) {
				if (m_displayLv>0)
					System.out.print("X");
				else
					System.out.println("X");
			}
			
			if (m_displayLv>0)
				System.out.println();
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
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_LinAdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
								
							userPerfStat = user.getPerfStat();								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									userPerfStat.addOnePredResult(predL, trueL);
								}
							}							
							userPerfStat.calculatePRF();	
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core, int numOfCores) {
					this.core = core;
					this.numOfCores = numOfCores;
					return this;
				}
			}).initialize(k, numberOfCores));
			
			threads.get(k).start();
		}
		
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
		
		int count = 0;
		double[] macroF1 = new double[m_classNo];
		_PerformanceStat userPerfStat;
		
		for(_LinAdaptStruct user:m_userList) {
			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
				continue;
			
			userPerfStat = user.getPerfStat();
			for(int i=0; i<m_classNo; i++)
				macroF1[i] += userPerfStat.getF1(i);
			m_microStat.accumulateConfusionMat(userPerfStat);
			count ++;
		}
		calcMicroPerfStat();
		
		// macro average
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++)
			System.out.format("Class %d: %.3f\t", i, macroF1[i]/count);
		System.out.println();
		return Utils.sumOfArray(macroF1);
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
		for(_LinAdaptStruct user:m_userList) {
			user.getPerfStat().clear();
		}			
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
