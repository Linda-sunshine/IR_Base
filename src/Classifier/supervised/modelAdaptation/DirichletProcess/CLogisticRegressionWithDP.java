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
	protected Normal m_normal; // Normal distribution.
	protected int m_M, m_kBar, m_count; // The number of auxiliary components.
	protected int m_numberOfIterations = 10;
	protected int m_burnIn = 10, m_thinning = 3;// burn in time, thinning time.
	protected double m_converge = -1e-9;
	protected double m_alpha = 1; // Scaling parameter of DP.
	protected double m_lambda = 1;
	
	// Parameters of the prior for the intercept and coefficients.
	protected double[] m_abNuA = new double[]{0, 1};
	protected double[] m_abNuB = new double[]{0, 1};
	protected double[] m_models; // model parameters for clusters.
	
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
	public int[] calculateClusetAssignment(){
		int index;
		int[] clusters = new int[m_kBar];
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			index = Arrays.asList(m_thetaStars).indexOf(user.getThetaStar());
			if(index > m_kBar-1)
				System.err.println("Cluster not found!");
			else
				clusters[index]++;
		}
		return clusters;
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
		double fValue = 0;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			fValue += calcLogLikelihood(user, 0);// 0 means we will use user's own thetastar.
		}
		return fValue;
	}
	protected double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_models.length; i++)
			R1 += m_lambda*m_models[i]*m_models[i];
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
		initLBFGS();
		init();
		assignClusterIndex();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient
				// Use instances inside one cluster to update the thetastar.
				for(int i=0; i<m_userList.size(); i++){
					user = (_DPAdaptStruct) m_userList.get(i);
					cIndex = user.getThetaStar().getIndex();
					fValue += calcLogLikelihood(user, cIndex);
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
		initThetaStars();
		// Burn in period.
		while(count++ < m_burnIn){
			calculate_E_step();
			calculate_M_step();
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
	// Return the indexes of the elements in arr which are larger than val.
	protected ArrayList<Integer> find(double[] arr, double val){
		ArrayList<Integer> res = new ArrayList<Integer>();
		for(int i=0; i<arr.length; i++){
			if(arr[i] > val)
				res.add(i);
		}
		return res;
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
	@Override
	protected int getVSize() {
		return m_kBar*m_dim;
	}
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n; // feature index
		int cIndex = Arrays.asList(m_thetaStars).indexOf(user.getThetaStar());
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
			m_g[i] += 2*m_lambda*m_models[i];
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
	// Generate a random vector.
	protected double[] normrnd(double[] us, double[] sigmas){
		if(us.length == 0 || sigmas.length == 0 || us.length != sigmas.length)
			return null;
		double[] rnds = new double[us.length];
		for(int i=0; i<us.length; i++){
			rnds[i] = m_normal.nextDouble(us[i], sigmas[i]);
		}
		return rnds;
	}
	// Sample one instance's cluster assignment.
	protected void sampleOneInstance(_DPAdaptStruct user){
		int picked;
		double u;
		double[] prob;
		sampleThetaStars(m_kBar, m_M);

		// Calculate the probability of each cluster.
		prob = new double[m_M+m_kBar];
		for(int k=0; k<m_kBar; k++){
			prob[k] = calcLogLikelihood(user, k);
			prob[k] += Math.log(m_thetaStars[k].getMemSize());
		}
		for(int m=0; m<m_M; m++){// new cluster
			prob[m+m_kBar] = calcLogLikelihood(user, m+m_kBar);
			prob[m+m_kBar] += Math.log(m_alpha) - Math.log(m_M);
		}
		normalizeProb(prob);
			
		// Pick one cluster assignment for the current instance.
		u = Math.random();
		picked = findIndex(prob, u); // picked is among [0,kBar+M].
		// Pick one cluster assignment for the current instance.
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
			m_thetaStars[start+m] = new _thetaStar(m_dim);
			m_thetaStars[start+m].setBeta(m_normal);
		}
	}

	// Set a bunch of parameters.
	public void setAlpha(double a){
		m_alpha = a;
	}
	public void setLambda(double lmd){
		m_lambda = lmd;
	}
	protected void setM(int m){
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
	protected double sumSquare(double[] beta){
		double val = 0;
		for(double b: beta)
			val += b*b;
		return val;
	}
	protected int m_test = 10;// We are trying to get the expectation of the performance.
	double[][] m_perfs;
	
	// In testing phase, we need to sample several times and get the average.
	@Override
	public double test(){
		m_perfs = new double[m_test][];
		m_M = 0;// We don't introduce new clusters.
		for(int i=0; i<m_test; i++){
			calculate_E_step();
			setPersonalizedModel();
			super.test();
			m_perfs[i] = Arrays.copyOf(m_perf, m_perf.length);
			clearPerformance();
		}
		double[] avgPerf = new double[m_classNo*2];
		System.out.print("[Info]Avg Perf:");
		for(int i=0; i<m_perfs[0].length; i++){
			avgPerf[i] = Utils.sumOfColumn(m_perfs, i)/m_test*1.0;
			System.out.print(String.format("%.5f\t", avgPerf[i]));
		}
		System.out.println();
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
		return String.format("CLRWithDP[dim:%d,M:%d,alpha:%.4f,lambda:%.2f,nuOfIter:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_M, m_alpha, m_lambda, m_numberOfIterations, m_eta1, m_eta2);
	}
	public void printInfo(){
		int[] clusters = calculateClusetAssignment();
		Arrays.sort(clusters);
		System.out.print("[Info]Clusters:");
		for(int i: clusters)
			System.out.print(i + "\t");
		System.out.println();
		System.out.print(String.format("[Info]%d Clusters are found in total!\n", getKBar()));
		
	}
}
