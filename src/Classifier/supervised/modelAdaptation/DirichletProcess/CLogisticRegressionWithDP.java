package Classifier.supervised.modelAdaptation.DirichletProcess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures._Doc;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt._DPAdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;

public class CLogisticRegressionWithDP extends LinAdapt{
	protected boolean m_burnInM = true; // Whether we have M step in burn in period.
	protected Normal m_normal; // Normal distribution.
	protected int m_M, m_kBar, m_count; // The number of auxiliary components.
	protected int m_numberOfIterations = 10;
	protected int m_burnIn = 10, m_thinning = 3;// burn in time, thinning time.
	protected double m_converge = 1e-9;
	protected double m_alpha = 1; // Scaling parameter of DP.
	
	// Parameters of the prior for the intercept and coefficients.
	protected double[] m_abNuA = new double[]{0, 10};
	protected double[] m_abNuB = new double[]{1, 1};
	protected double[] m_models; // model parameters for clusters.
	protected double[] m_probs;
	_thetaStar[] m_thetaStars = new _thetaStar[1000];
//	HashMap<_thetaStar, Integer> m_thetaStarMap;
	public CLogisticRegressionWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel){
		super(classNo, featureSize, featureMap, globalModel);
		m_dim = m_featureSize+1;
		m_M = 5; 
		m_kBar = 0;// Initial value, assigned to one cluster.
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	protected void accumulateClusterModels(){
		m_models = new double[getVSize()];
		for(int i=0; i<m_kBar; i++){
			System.arraycopy(m_thetaStars[i].m_beta, 0, m_models, m_dim*i, m_dim);
		}
	}
	protected void assignClusterIndex(){
		for(int i=0; i<m_kBar; i++)
			m_thetaStars[i].setIndex(i);
	}
	//Calculate the function value of the new added instance.
	protected double calcLogLikelihood(_AdaptStruct user, int k){
		double L = 0; //log likelihood.
		double Pi = 0;
		
		for(_Review review:user.getReviews()){
			if (review.getType() != rType.ADAPTATION)
				continue; // only touch the adaptation data
			
			Pi = logit(review.getSparse(), user, k);
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
		if(m_LNormFlag)
			return L/getAdaptationSize(user);
		else
			return L;
	}
	// Calcualte loglikelihood after get the weights.
	protected double calcLoglikelihood(){
		_DPAdaptStruct user;
		double L = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			L += calcLogLikelihood(user, user.getThetaStar().getIndex());
		}
		return L;
	}
	// After we fix the clusters, we calculate the probability each user belongs to each cluster.
	public void calculateClusterProb(){
		double prob;
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			user.setThetaStars(m_thetaStars);
			for(int k=0; k<m_kBar; k++){
				prob = calcLogLikelihood(user, k);
				prob += Math.log(m_thetaStars[k].getMemSize());
				m_probs[k] = Math.exp(prob);
			}
			Utils.L1Normalization(m_probs);
			user.setCProb(m_probs);
		}
	}
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_models.length; i++)
			R1 += (m_models[i]-m_abNuA[0])*(m_models[i]-m_abNuA[0])/(m_abNuA[1]*m_abNuA[1]);
		return R1;
	}
	// The main MCMC algorithm, assign each user to clusters.
	protected void calculate_E_step(){
		int cIndex;
		_thetaStar curThetaStar;
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			curThetaStar = user.getThetaStar();
			curThetaStar.memSizeMinusOne();
			if(curThetaStar.getMemSize() == 0){// No data associated with the cluster.
				cIndex = Arrays.asList(m_thetaStars).indexOf(curThetaStar);
				m_thetaStars[cIndex] = m_thetaStars[m_kBar-1]; // Use the last thetastar to cover this one.
				m_kBar--;// kBar starts from 0, the size decides how many are valid.
			}
			sampleOneInstance(user);
		}
	}
	// Sample the weights given the cluster assignment.
	protected void calculate_M_step(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0, cIndex;
		_DPAdaptStruct user;
		
		initLBFGS();// Init for lbfgs.
		assignClusterIndex();
		try{
			do{
				fValue = 0;
				initPerIter();
				// Use instances inside one cluster to update the thetastar.
				for(int i=0; i<m_userList.size(); i++){
					user = (_DPAdaptStruct) m_userList.get(i);
					cIndex = user.getThetaStar().getIndex();
					fValue -= calcLogLikelihood(user, cIndex);
					gradientByFunc(user); // calculate the gradient by the user.
				}
				accumulateClusterModels();
				fValue += calculateR1();
				gradientByR1();
				if (m_displayLv==2) {
					gradientTest();
					System.out.print("Fvalue is " + fValue + "\t");
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
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}		
		setPersonalizedModel();
	}	
	// Accumulate the sum the elements of the array.
	protected void cumsum(double[] arr){
		double sum = 0;
		for(int i=0; i<arr.length; i++){
			sum += arr[i];
			arr[i] = sum;
		}
	}
	// Accumulate the sum the elements of the array.
	protected double cumLogSum(){
		double sum = 0, max = Integer.MIN_VALUE, prob;
		// Find the max value.
		for(int i=0; i<m_kBar+m_M; i++){
			prob = m_thetaStars[i].getProb();
			if(prob > max)
				max = prob;
		}
		for (int i=0; i<m_kBar+m_M; i++){
			prob = m_thetaStars[i].getProb();
			sum += Math.exp(prob - max);
		}
		if(sum == 0)
			return max;
		return Math.log(sum) + max;
	}
	// Use different weights for dot product.
	protected double dotProduct(_SparseFeature[] fvs, double[] weights){
		double sum = weights[0]; // bias term
		for(_SparseFeature f:fvs) 
			sum += weights[f.getIndex()+1] * f.getValue();		
		return sum;
	}
	
	// The main EM algorithm to optimize cluster assignment and distribution parameters.
	public void EM(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		int count = 0;
		
		initThetaStars();// Init cluster assignment.
		init(); // clear user performance.
		
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			if(m_burnInM){
				calculate_M_step();
			}
		}
		// EM iteration.
		for(int i=0; i<m_numberOfIterations; i++){
			// Cluster assignment, thinning to reduce auto-correlation.
			for(int j=0; j<m_thinning; j++)
				calculate_E_step();
			// Optimize the parameters.
			calculate_M_step();
			lastLikelihood = curLikelihood;
			curLikelihood = calcLoglikelihood();

			delta = curLikelihood - lastLikelihood;
			System.out.print(String.format("[Info]Step %d: Delta_likelihood: %.3f\n", i, delta));
			if(Math.abs(delta) < m_converge)
				break;
		}
	}
	// Find the index of region the picked probability falls in.
	protected int findIndex(double[] prob, double u){
		int start = 0, end = prob.length-1;
		if(u < prob[0] && u >=0)
			return 0;
		if(u > prob[prob.length-2] && u <= 1)
			return prob.length-1;
		int mid = (start+end)/2;
		while(!(u<=prob[mid] && u>prob[mid-1])){
			if(u > prob[mid])
				start = mid;
			else
				end = mid;
			mid = (start+end)/2;
		}
		return mid;
	}
	// Find the index of a specific value.
	protected int findIndex(double sum){
		double subsum = 0, u = Math.random() * sum;
		int picked = 0;
		subsum = m_thetaStars[0].getProb();
		if(u <= subsum)
			return 0;
		for(int i=1; i<m_kBar+m_M; i++){
			subsum = Math.log(Math.exp(subsum)+Math.exp(m_thetaStars[i].getProb()));
			if(u <= subsum){
				picked = i;
				break;
			}
		}
		return picked;
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
		double delta = (review.getYLabel() - logit(review.getSparse(), user, cIndex));
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		//Bias term.
		m_g[offset] -= weight*delta; //x0=1

		//Traverse all the feature dimension to calculate the gradient.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			m_g[offset + n] -= weight * delta * fv.getValue();
		}
	}
	// Gradient by the regularization.
	protected void gradientByR1(){
		for(int i=0; i<m_g.length; i++)
			m_g[i] += 2*(m_models[i]-m_abNuA[0])/(m_abNuA[1]*m_abNuA[1]);
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
	public int getKBar(){
		return m_kBar;
	}
	// Assign cluster assignment to each user.
	protected void initThetaStars(){
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			sampleOneInstance(user);
		}
	}
	@Override
	protected void initLBFGS(){
		m_g = new double[getVSize()];
		m_diag = new double[getVSize()];
		Arrays.fill(m_g, 0);
		Arrays.fill(m_diag, 0);
	}
	// Init in each iteration in M step.
	protected void initPerIter() {
		Arrays.fill(m_g, 0); // initialize gradient
		Arrays.fill(m_diag, 0);
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user));
		m_pWeights = new double[m_gWeights.length];		
	}
	// Logit function is different from the father class.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u, int k){
		_thetaStar curThetaStar = m_thetaStars[k];			
		double sum = dotProduct(fvs, curThetaStar.m_beta);
		return Utils.logistic(sum);
	}
	// Normalize the probability to sum up to one.
	protected void normalizeProb(double[] prob){
		double maxP = Utils.maxOfArrayValue(prob);
		for(int j=0; j<prob.length; j++){
			prob[j] = Math.exp(prob[j]-maxP);
		}
		Utils.L1Normalization(prob);
		cumsum(prob);
	}
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		int picked;
		double prob, sum;
		sampleThetaStars(m_kBar, m_M);
		
		double[] probs = new double[m_kBar+m_M];
		for(int k=0; k<m_kBar; k++){
			prob = calcLogLikelihood(user, k);
			prob += Math.log(m_thetaStars[k].getMemSize());
			m_thetaStars[k].setProb(prob);
			probs[k] = prob;
		}
		for(int m=0; m<m_M; m++){// new cluster
			prob = calcLogLikelihood(user, m+m_kBar);
			prob += Math.log(m_alpha) - Math.log(m_M);
			m_thetaStars[m+m_kBar].setProb(prob);
			probs[m_kBar+m] = prob;
		}
//		sum = cumLogSum();
		normalizeProb(probs);
		// Pick one cluster assignment for the current instance.
		
		picked = findIndex(probs, Math.random()); // picked is among [0,kBar+M].
		m_thetaStars[picked].memSizeAddOne();
		user.setThetaStar(m_thetaStars[picked]);
		if(picked >= m_kBar){
			m_thetaStars[m_kBar] = m_thetaStars[picked];
			m_kBar++;
		}
	}
	// Sample thetaStars.
	protected void sampleThetaStars(int start, int M){
		for(int m=0; m<M; m++){
			if(m_thetaStars[start+m] == null)
				m_thetaStars[start+m] = new _thetaStar(m_dim, m_abNuA);
			m_thetaStars[start+m].setBeta(m_normal);
		}
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
			user.setPersonalizedModel(user.getThetaStar().m_beta);
		}
	}
	// Assign the optimized weights to the cluster.
	protected void setThetaStars(){
		for(int i=0; i<m_kBar; i++){
			System.arraycopy(m_models, m_dim*i, m_thetaStars[i].m_beta, 0, m_dim);
		}
	}
	protected int m_test = 10;// We are trying to get the expectation of the performance.
	double[][] m_perfs;
	
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
	@Override
	public double test(){
		// we sample one time.
		calculate_E_step();
		// we calculate the user's cluster probability.
		m_probs = new double[m_kBar];
		calculateClusterProb();
//		setPersonalizedModel();
		super.test();	
		return 0;
	}
	
	protected void clearPerformance(){
		Arrays.fill(m_perf, 0);
		m_microStat.clear();
		for(_AdaptStruct u: m_userList)
			u.getUser().getPerfStat().clear();
	}
	@Override
	public String toString() {
		return String.format("CLRWithDP[dim:%d,M:%d,alpha:%.4f,nuOfIter:%d]", m_dim, m_M, m_alpha, m_numberOfIterations);
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
		System.out.print(String.format("[Info]%d Clusters are found in total!\n", getKBar()));
		
	}
}
