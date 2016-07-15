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
/***
 * This is a translation of dpMnl from matlab to java.
 * "Nonlinear models using dirichlet process mixtures"
 * @author lin
 */
public class MultiTaskWithDP extends LinAdapt{
	Normal m_normal; // Normal distribution.
	int m_M, m_kBar, m_count; // The number of auxiliary components.
	int m_numberOfIterations = 10;
	
	double m_converge = -1e-9;
	double m_alpha = 0.0001; // Scaling parameter of DP.
	double m_lambda = 10;
	
	// Parameters of the prior for the intercept and coefficients.
	double[] m_abNuA = new double[]{0, 1};
	double[] m_abNuB = new double[]{0, 1};
	double[] m_weights; // weights for clusters.
	
	_thetaStar[] m_thetaStars = new _thetaStar[1000];
//	HashMap<_thetaStar, Integer> m_thetaStarMap;
	public MultiTaskWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_dim = m_featureSize+1;
		m_M = 5; 
		m_kBar = 0;// Initial value, assigned to one cluster.
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	public void accumulateClusterWeights(){
		m_weights = new double[getVSize()];
		for(int i=0; i<m_kBar; i++){
			System.arraycopy(m_thetaStars[i].m_beta, 0, m_weights, m_dim*i, m_dim);
		}
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
	public double calculateR1(){
		double R1 = 0;
		for(int i=0; i<m_weights.length; i++)
			R1 += m_lambda*m_weights[i]*m_weights[i];
		return R1;
	}
	// The main MCMC algorithm, assign each user to clusters.
	public void calculate_E_step(){
		double[] prob;
		double u;
		int picked, cIndex;
		_thetaStar curThetaStar;
		for(int i=0; i<m_userList.size(); i++){
			_DPAdaptStruct user = (_DPAdaptStruct) m_userList.get(i);
			curThetaStar = user.getThetaStar();
			curThetaStar.memSizeMinusOne();
			if(curThetaStar.getMemSize() == 0){// No data associated with the cluster.
				cIndex = Arrays.asList(m_thetaStars).indexOf(curThetaStar);
				m_thetaStars[cIndex] = m_thetaStars[m_kBar-1]; // Use the last thetastar to cover this one.
				m_kBar--;// kBar starts from 0, the size decides how many are valid.
			}
			sampleThetaStars(m_kBar, m_M); // sample new thetaStars.
	
			// Calculate the probability of each cluster.
			prob = new double[m_kBar+m_M];
			for(int k=0; k<m_kBar; k++){// existing cluster
				prob[k] = calcLogLikelihood(user);
				prob[k] += Math.log(m_thetaStars[k].getMemSize());
			}
			for(int m=0; m<m_M; m++){// new cluster
				prob[m+m_kBar] = calcLogLikelihood(user, m_kBar+m);
				prob[m+m_kBar] += Math.log(m_alpha) - Math.log(m_M);
			}
			normalizeProb(prob);
				
			// Pick one cluster assignment for the current instance.
			u = Math.random();
			picked = findIndex(prob, u); // picked is among [0,kBar+M].
			m_thetaStars[picked].memSizeAddOne();
			user.setThetaStar(m_thetaStars[picked]);
			if(picked >= m_kBar){
				m_thetaStars[m_kBar] = m_thetaStars[picked];
				m_kBar++;
			}
		}
	}
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
				LBFGS.lbfgs(m_g.length, 5, m_weights, fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
				setThetaStars();
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}		
		setPersonalizedModel();
	}	

//	public double calcLogPostSigma(double[] beta, double x, double[] param){
//		double sigma = Math.sqrt(Math.log(x));
//		double logPost = -beta.length*(Math.log(Math.sqrt(2*Math.PI*sigma*sigma)))-sumSquare(beta)/(2*sigma*sigma)
//						 -Math.log(Math.sqrt(Math.PI*2*param[0]*param[0]))-(x-param[0])*(x-param[0])/(2*param[1]*param[1]);
//		return logPost;
//	}
//	public double calcLogPostBeta(int uIndex, double[][] beta, double[][] sigma){
//		double logLike = calcLogLikelihood(m_userList.get(uIndex), m_J[uIndex]);
//		double logPrior = calcPrior(beta, sigma);
//		return logLike + logPrior;
//	}
//	// beta ~ N(0, sigma)
//	public double calcPrior(double[][] beta, double[][] sigma){
//		double val = 0;
//		for(int i=0;i < beta.length; i++){
//			for(int j=0; j<beta[0].length; j++){
//				val += -0.5*beta[i][j]*beta[i][j]/(sigma[i][j]*sigma[i][j]);
//			}
//		}
//		return val;
//	}

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
	public void EM(){
		System.out.println(toString());
		double delta = 0, lastLikelihood = 0, curLikelihood = 0;
		for(int i=0; i<m_numberOfIterations; i++){
			calculate_E_step();
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
	public ArrayList<Integer> find(double[] arr, double val){
		ArrayList<Integer> res = new ArrayList<Integer>();
		for(int i=0; i<arr.length; i++){
			if(arr[i] > val)
				res.add(i);
		}
		return res;
	}
	// Find the index of region the picked probability falls in.
	public int findIndex(double[] prob, double u){
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
	public int getKBar(){
		return m_kBar;
	}
	// Define the first thetaStar and assign all users to the cluster.
	public void initThetaStars(){
		m_thetaStars[0] = new _thetaStar(m_dim);
		m_thetaStars[0].setBeta(m_normal);
		m_kBar++;
		// Assign thetaStar to the users.
		_DPAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			user.setThetaStar(m_thetaStars[0]);
			m_thetaStars[0].memSizeAddOne();
		}
	}
	@Override
	public void initLBFGS(){
		m_g = new double[getVSize()];
		m_diag = new double[getVSize()];
		Arrays.fill(m_g, 0);
		Arrays.fill(m_diag, 0);
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user, m_dim));
		m_pWeights = new double[m_gWeights.length];		
		initThetaStars();
	}

	// Logit function is different from the father class.
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u, int k){
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		_thetaStar curThetaStar;
		if(k <= m_kBar)
			curThetaStar = user.getThetaStar();// existing cluster.
		else
			curThetaStar = m_thetaStars[k];					
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

	// Sample thetaStars.
	public void sampleThetaStars(int start, int M){
		for(int m=0; m<M; m++){
			m_thetaStars[start+m] = new _thetaStar(m_dim);
			m_thetaStars[start+m].m_nuA = m_normal.nextDouble(m_abNuA[0], m_abNuA[1]);
			m_thetaStars[start+m].m_nuB = m_normal.nextDouble(m_abNuB[0], m_abNuB[1]);
			m_thetaStars[start+m].setBeta(m_normal);
		}
	}
//	public double sampleSigBeta(double[] beta, double sigma2, double[] param){
//		double x = Math.log(sigma2);
//		double z = calcLogPostSigma(beta, x, param);
//		return Math.exp(z);
//	}
//	// Sample the cluster weights with fixed cluster assignment.
//	public double[][] sampleBeta(int uIndex, double[][] beta, double[][] sigma){
//		double postBeta = calcLogPostBeta(uIndex, beta, sigma);
//		
//	}

	// Set a bunch of parameters.
	public void setM(int m){
		m_M = m;
	}
	public void setLambda(double lmd){
		m_lambda = lmd;
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
	// Sqrt+Exp+Normalize for an array from normal distributions parameterized by values in the two parameters.
//	public double[] sqrtExpNormrnd(double[] us, double[] sigmas){
//		if(us.length == 0 || sigmas.length == 0 || us.length != sigmas.length)
//			return null;
//		double[] rnds = new double[us.length];
//		for(int i=0; i<us.length; i++){
//			rnds[i] = sqrtExpNormrndOne(us[i], sigmas[i]);
//		}
//		return rnds;
//	}
//	// Sqrt+Exp+Normalize for one random value.
//	public double sqrtExpNormrndOne(double u, double sigma){
//		return Math.sqrt(Math.exp(m_normal.nextDouble(u, sigma)));
//	}
	
//	public int[] unique(int[] arr){
//		HashSet<Integer> set = new HashSet<Integer>();
//		for(int i: arr)
//			set.add(i);
//		ArrayList<Integer> list = new ArrayList<Integer>(set);
//		Collections.sort(list);
//		int[] res = new int[list.size()];
//		for(int i=0; i<list.size(); i++)
//			res[i] = list.get(i);
//		return res;
//	} 
	@Override
	public String toString() {
		return String.format("MultiTaskWithDP[dim:%d,M:%d,eta1:%.3f,eta2:%.3f]", m_dim, m_M, m_eta1, m_eta2);
	}
	
	public static void main(String[] args){
		double[] p = new double[]{0.1, 0.3, 0.5, 0.75, 0.88, 1};
		MultiTaskWithDP test = new MultiTaskWithDP(0, 0, null, "", "");
		System.out.println(test.findIndex(p, 1));	
	}
}
