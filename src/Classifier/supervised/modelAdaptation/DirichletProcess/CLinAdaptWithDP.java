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
import Classifier.supervised.modelAdaptation.CoLinAdapt._DPAdaptStruct;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
/***
 * This is a translation of dpMnl from matlab to java.
 * "Nonlinear models using dirichlet process mixtures"
 * @author lin
 */
public class CLinAdaptWithDP extends CLogisticRegressionWithDP{
	
	public CLinAdaptWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize, featureMap, globalModel);
		loadFeatureGroupMap(featureGroupMap);
		m_M = 10; 
		m_kBar = 0;// Initial value, assigned to one cluster.
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	public void accumulateClusterWeights(){
		m_weights = new double[getVSize()];
		for(int i=0; i<m_kBar; i++){
			System.arraycopy(m_thetaStars[i].m_beta, 0, m_weights, m_dim*i, m_dim);
		}
	}
	public int[] calculateCluserAssignment(){
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
	public double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_weights.length; i++)
			R1 += m_lambda*m_weights[i]*m_weights[i];
		return R1;
	}
	// The main MCMC algorithm, assign each user to clusters.
//	public void calculate_E_step(){
//		int cIndex;
//		_thetaStar curThetaStar;
//		_DPAdaptStruct user;
//		for(int i=0; i<m_userList.size(); i++){
//			user = (_DPAdaptStruct) m_userList.get(i);
//			curThetaStar = user.getThetaStar();
//			curThetaStar.memSizeMinusOne();
//			if(curThetaStar.getMemSize() == 0){// No data associated with the cluster.
//				cIndex = Arrays.asList(m_thetaStars).indexOf(curThetaStar);
//				m_thetaStars[cIndex] = m_thetaStars[m_kBar-1]; // Use the last thetastar to cover this one.
//				m_kBar--;// kBar starts from 0, the size decides how many are valid.
//			}
//			sampleOneInstance(user);
//		}
//	}
	// Sample the weights given the cluster assignment.
	public void calculate_M_step(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;
		int displayCount = 0;
		_DPAdaptStruct user;
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient
				// Use instances inside one cluster to update the thetastar.
				for(int i=0; i<m_userList.size(); i++){
					user = (_DPAdaptStruct) m_userList.get(i);
					fValue += calcLogLikelihood(user, 0);// 0 means we will use user's own thetastar.
					gradientByFunc(user); // calculate the gradient by the user.
				}
				accumulateClusterWeights();
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
				LBFGS.lbfgs(m_g.length, 5, m_weights, fValue, m_g, false, m_diag, iprint, 1e-2, 1e-16, iflag);//In the training process, A is updated.
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
	public void cumsum(double[] arr){
		double sum = 0;
		for(int i=0; i<arr.length; i++){
			sum += arr[i];
			arr[i] = sum;
		}
	}
	// Use different weights for dot product.
	public double dotProduct(_SparseFeature[] fvs, double[] weights){
		double sum = weights[0]; // bias term
		for(_SparseFeature f:fvs) 
			sum += weights[f.getIndex()+1] * f.getValue();		
		return Utils.logistic(sum);
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
	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2;
	}
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n; // feature index
		int cIndex = Arrays.asList(m_thetaStars).indexOf(user.getThetaStar());
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		int offset = m_dim*cIndex;
		double delta = (review.getYLabel() - logit(review.getSparse(), user, 0));
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
	public void gradientByR1(){
		for(int i=0; i<m_g.length; i++)
			m_g[i] += 2*m_lambda*m_weights[i];
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

	// Assign cluster assignment to each user.
	public void initThetaStars(){
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			sampleOneInstance(user);
		}
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user, m_dim));
		m_pWeights = new double[m_gWeights.length];		
	}
	// Logit function is different from the father class.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u, int k){
		_thetaStar curThetaStar = m_thetaStars[k];					
		double sum = dotProduct(fvs, curThetaStar.m_beta);// new cluster.	
		return Utils.logistic(sum);
	}
	// Normalize the probability to sum up to one.
	public void normalizeProb(double[] prob){
		double maxP = Utils.maxOfArrayValue(prob);
		for(int j=0; j<prob.length; j++){
			prob[j] = Math.exp(prob[j]-maxP);
		}
		Utils.L1Normalization(prob);
		cumsum(prob);
	}
	// Generate a random vector.
	public double[] normrnd(double[] us, double[] sigmas){
		if(us.length == 0 || sigmas.length == 0 || us.length != sigmas.length)
			return null;
		double[] rnds = new double[us.length];
		for(int i=0; i<us.length; i++){
			rnds[i] = m_normal.nextDouble(us[i], sigmas[i]);
		}
		return rnds;
	}
	// Sample one instance's cluster assignment.
	public void sampleOneInstance(_DPAdaptStruct user){
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
	public void sampleThetaStars(int start, int M){
		for(int m=0; m<M; m++){
			m_thetaStars[start+m] = new _thetaStar(m_dim);
			m_thetaStars[start+m].m_nuA = m_normal.nextDouble(m_abNuA[0], m_abNuA[1]);
			m_thetaStars[start+m].m_nuB = m_normal.nextDouble(m_abNuB[0], m_abNuB[1]);
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
	public void setM(int m){
		m_M = m;
	}
	public void setNumberOfIterations(int num){
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
			System.arraycopy(m_weights, m_dim*i, m_thetaStars[i].m_beta, 0, m_dim);
		}
	}
	public double sumSquare(double[] beta){
		double val = 0;
		for(double b: beta)
			val += b*b;
		return val;
	}
	int m_test = 10;// We are trying to get the expectation of the performance.
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
		for(int i=0; i<m_perfs[0].length; i++)
			System.out.print(Utils.sumOfColumn(m_perfs, i)/m_test*1.0+"\t");
		System.out.println();
		return 0;
	}
	public void clearPerformance(){
		Arrays.fill(m_perf, 0);
		m_microStat.clear();
		for(_AdaptStruct u: m_userList)
			u.getUser().getPerfStat().clear();
	}
	@Override
	public String toString() {
		return String.format("MultiTaskWithDP[dim:%d,M:%d,alpha:%.4f,eta1:%.3f,eta2:%.3f]", m_dim, m_M, m_alpha, m_eta1, m_eta2);
	}
	
	public static void main(String[] args){
		double[] p = new double[]{0.1, 0.3, 0.5, 0.75, 0.88, 1};
		CLinAdaptWithDP test = new CLinAdaptWithDP(0, 0, null, "", "");
		System.out.println(test.findIndex(p, 1));	
	}
}
