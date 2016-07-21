package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import cern.jet.random.tfloat.FloatUniform;
import structures._Doc;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import utils.Utils;

public class CLogisticRegressionWithDP extends LinAdapt {
	protected boolean m_burnInM = false; // Whether we have M step in burn in period.
	
	protected int m_M = 5, m_kBar = 0; // The number of auxiliary components.
	protected int m_numberOfIterations = 15;
	protected int m_burnIn = 10, m_thinning = 3;// burn in time, thinning time.
	protected double m_converge = 1e-9;
	protected double m_alpha = 1; // Scaling parameter of DP.
	protected double m_pNewCluster; // proportion of sampling a new cluster, to be assigned before EM starts
	protected NormalPrior m_G0; // prior distribution
	
	// Parameters of the prior for the intercept and coefficients.
	protected double[] m_abNuA = new double[]{0, 0.3}; // N(0,1) for shifting
	protected double[] m_models; // model parameters for clusters.
	public static _thetaStar[] m_thetaStars = new _thetaStar[1000];//to facilitate prediction in each user 

	public CLogisticRegressionWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel){
		super(classNo, featureSize, featureMap, globalModel, null);
		m_dim = m_featureSize + 1; // to add the bias term
	}
	
	protected void assignClusterIndex(){
		for(int i=0; i<m_kBar; i++)
			m_thetaStars[i].setIndex(i);
	}
	
	// After we finish estimating the clusters, we calculate the probability of each user belongs to each cluster.
	protected void calculateClusterProbPerUser(){
		double prob;
		_DPAdaptStruct user;
		double[] probs = new double[m_kBar];
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			for(int k=0; k<m_kBar; k++){
				user.setThetaStar(m_thetaStars[k]);
				prob = calcLogLikelihood(user) + Math.log(m_thetaStars[k].getMemSize());//this proportion includes the user's current cluster assignment
				probs[k] = Math.exp(prob);//this will be in real space!
			}
			Utils.L1Normalization(probs);
			user.setClusterPosterior(probs);
		}
	}
	
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_kBar; i++)
			R1 += m_G0.likelihood(m_thetaStars[i].getModel());
		return R1;
	}
	
	int findThetaStar(_thetaStar theta) {
		for(int i=0; i<m_kBar; i++)
			if (theta == m_thetaStars[i])
				return i;
		return -1;// impossible to hit here!
	}
	
	// Sample thetaStars.
	protected void sampleThetaStars(){
		for(int m=m_kBar; m<m_kBar+m_M; m++){
			if (m_thetaStars[m] == null) {
				if (this instanceof CLinAdaptWithDP)
					m_thetaStars[m] = new _thetaStar(2*m_dim);
				else
					m_thetaStars[m] = new _thetaStar(m_dim);
			}
			m_G0.sampling(m_thetaStars[m].getModel());
		}
	}
	
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		double likelihood, logSum = 0;
		int k;
		
		//reset thetaStars
		sampleThetaStars();
		for(k=0; k<m_kBar+m_M; k++){
			user.setThetaStar(m_thetaStars[k]);
			likelihood = calcLogLikelihood(user);
			
			if (k<m_kBar)
				likelihood += Math.log(m_thetaStars[k].getMemSize());
			else
				likelihood += m_pNewCluster;
			 
			m_thetaStars[k].setProportion(likelihood);//this is in log space!
			
			if (k==0)
				logSum = likelihood;
			else
				logSum = Utils.logSum(logSum, likelihood);
		}
		
		logSum += Math.log(FloatUniform.staticNextFloat());//we might need a better random number generator
		
		k = 0;
		double newLogSum = m_thetaStars[0].getProportion();
		do {
			if (newLogSum>=logSum)
				break;
			k++;
			newLogSum = Utils.logSum(newLogSum, m_thetaStars[k].getProportion());
		} while (k<m_kBar+m_M);
		
		if (k==m_kBar+m_M)
			k--; // we might hit the very last
		
		m_thetaStars[k].updateMemCount(1);
		user.setThetaStar(m_thetaStars[k]);
		if(k >= m_kBar){
			swapTheta(m_kBar, k);
			m_kBar++;
		}
	}	
	
	void swapTheta(int a, int b) {
		_thetaStar cTheta = m_thetaStars[a];
		m_thetaStars[a] = m_thetaStars[b];
		m_thetaStars[b] = cTheta;// kBar starts from 0, the size decides how many are valid.
	}
	
	// The main MCMC algorithm, assign each user to clusters.
	protected void calculate_E_step(){
		_thetaStar curThetaStar;
		_DPAdaptStruct user;
		
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			curThetaStar = user.getThetaStar();
			curThetaStar.updateMemCount(-1);
			
			if(curThetaStar.getMemSize() == 0) {// No data associated with the cluster.
				swapTheta(m_kBar-1, findThetaStar(curThetaStar)); // move it back to \theta*
				m_kBar --;
			}
			sampleOneInstance(user);
		}
	}
	
	// Sample the weights given the cluster assignment.
	protected double calculate_M_step(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0;
		_DPAdaptStruct user;
		
		initLBFGS();// Init for lbfgs.
		assignClusterIndex();
		
		try{
			do{
				Arrays.fill(m_g, 0); // initialize gradient
				
				fValue = calculateR1();				
				// Use instances inside one cluster to update the thetastar.
				for(int i=0; i<m_userList.size(); i++){
					user = (_DPAdaptStruct) m_userList.get(i);
					fValue -= calcLogLikelihood(user);
					gradientByFunc(user); // calculate the gradient by the user.
				}				
				
				gradientByR1();
				
				if (m_displayLv==2) {
					System.out.print("Fvalue is " + fValue + "\t");
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				
				LBFGS.lbfgs(m_g.length, 5, m_models, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				setThetaStars();
				oldFValue = fValue;
				
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}	
		return oldFValue;
	}
	
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	public double train(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		init(); // clear user performance and init cluster assignment		
		
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			if(m_burnInM)
				calculate_M_step();
		}
		
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			for(int j=0; j<m_thinning; j++)
				calculate_E_step();
			
			// Optimize the parameters
			curLikelihood = calculate_M_step();

			delta = curLikelihood - lastLikelihood;
			
			printInfo();
			System.out.print(String.format("[Info]Step %d: likelihood: %.4f, Delta_likelihood: %.3f\n", i, curLikelihood, delta));
			if(Math.abs(delta) < m_converge)
				break;
			lastLikelihood = curLikelihood;
		}
//		setPersonalizedModel();
		return curLikelihood;
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		
		int offset = m_dim*cIndex;
		double delta = (review.getYLabel() - logit(review.getSparse(), user));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[offset] -= weight * delta; //x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= weight * delta * fv.getValue();
		}
	}
	
	// Gradient by the regularization.
	protected void gradientByR1(){
		for(int i=0; i<m_g.length; i++)
			m_g[i] += (m_models[i]-m_abNuA[0]) / (m_abNuA[1]*m_abNuA[1]);
	}
	
	@Override
	protected double gradientTest() {
		double mag = 0 ;
		for(int i=0; i<m_g.length; i++)
			mag += m_g[i]*m_g[i];

		if (m_displayLv==2)
			System.out.format("Gradient magnitude: %.5f\n", mag);
		return mag;
	}
	
	protected void initPriorG0() {
		m_G0 = new NormalPrior(m_abNuA[0], m_abNuA[1]);//only for shifting
	}
	
	// Assign cluster assignment to each user.
	protected void initThetaStars(){
		initPriorG0();
		m_pNewCluster = Math.log(m_alpha) - Math.log(m_M);//to avoid repeated computation
		
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			sampleOneInstance(user);
		}		
	}
	
	@Override
	protected void init(){
		super.init();
		initThetaStars();
	}
	
	protected void accumulateClusterModels(){
		m_models = new double[getVSize()];
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_thetaStars[i].getModel(), 0, m_models, m_dim*i, m_dim);
	}
	
	//very inefficient, a per cluster optimization procedure will not have this problem
	@Override
	protected void initLBFGS(){
		m_g = new double[getVSize()];
		m_diag = new double[getVSize()];
		Arrays.fill(m_g, 0);
		Arrays.fill(m_diag, 0);
		
		accumulateClusterModels();
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	
	@Override	
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		double sum = Utils.dotProduct(((_DPAdaptStruct)u).getThetaStar().getModel(), fvs, 0);
		return Utils.logistic(sum);
	}

	// Set a bunch of parameters.
	public void setAlpha(double a){
		m_alpha = a;
	}
	
	public void setBurnIn(int n){
		m_burnIn = n;
	}
	
	public void setBurnInM(boolean b){
		m_burnInM = b;
	}
	
	public void setM(int m){
		m_M = m;
	}
	
	protected void setNumberOfIterations(int num){
		m_numberOfIterations = num;
	}
	
	@Override
	protected void setPersonalizedModel() {
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			user.setPersonalizedModel(user.getThetaStar().getModel());
		}
	}
	
	// Assign the optimized weights to the cluster.
	protected void setThetaStars(){
		double[] beta;
		for(int i=0; i<m_kBar; i++){
			beta = m_thetaStars[i].getModel();
			System.arraycopy(m_models, i*m_dim, beta, 0, m_dim);
		}
	}
	
//	protected int m_test = 10;// We are trying to get the expectation of the performance.
//	double[][] m_perfs;
//	
	// In testing phase, we need to sample several times and get the average.
//	@Override
//	public double test(){
//		m_perfs = new double[m_test][];
//		m_M = 0;// We don't introduce new clusters.
//		for(int i=0; i<m_test; i++){
//			calculate_E_step();
//			setPersonalizedModel();
//			super.test();
//			m_perfs[i] = Arrays.copyOf(m_perf, m_perf.length);
//			clearPerformance();
//		}
//		double[] avgPerf = new double[m_classNo*2];
//		System.out.print("[Info]Avg Perf:");
//		for(int i=0; i<m_perfs[0].length; i++){
//			avgPerf[i] = Utils.sumOfColumn(m_perfs, i)/m_test*1.0;
//			System.out.print(String.format("%.5f\t", avgPerf[i]));
//		}
//		System.out.println();
//		return 0;
//	}
//	
//	protected void clearPerformance(){
//		Arrays.fill(m_perf, 0);
//		m_microStat.clear();
//		for(_AdaptStruct u: m_userList)
//			u.getUser().getPerfStat().clear();
//	}
	
	@Override
	public double test(){
		// we calculate the user's cluster probability.
		calculateClusterProbPerUser();
		return super.test();	
	}
	
	@Override
	public String toString() {
		return String.format("CLRegWithDP[dim:%d,M:%d,alpha:%.4f,#Iter:%d,N(%.3f,%.3f)]", m_dim, m_M, m_alpha, m_numberOfIterations, m_abNuA[0], m_abNuA[1]);
	}
	
	public void printInfo(){
		int[] clusters = new int[m_kBar];
		for(int i=0; i<m_kBar; i++)
			clusters[i] = m_thetaStars[i].getMemSize();
		Arrays.sort(clusters);
		System.out.print("[Info]Clusters:");
		for(int i: clusters)
			System.out.print(i+"\t");
		System.out.println();
		System.out.print(String.format("[Info]%d Clusters are found in total!\n", m_kBar));
	}
}
