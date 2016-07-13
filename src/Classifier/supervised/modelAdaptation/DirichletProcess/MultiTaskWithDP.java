package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.CoLinAdapt.LinAdapt;
import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
/***
 * This is a translation of dpMnl from matlab to java.
 * "Nonlinear models using dirichlet process mixtures"
 * @author lin
 */
public class MultiTaskWithDP extends LinAdapt{
	Normal m_normal; // Normal distribution.
	int m_M, m_kBar; // The number of auxiliary components.
	int m_numberOfIterations;
	
	double m_eps = 0.2; // This is the constant multiplier for the step size.
//	double m_a0 =-3, m_b0 = 2; // alpha~Gamma(a0, b0)
	double m_alpha = 0.001; // Scale parameter of DP.
	
	int[] m_J; // cluster assignment of users.
	TreeMap<Integer,  ArrayList<Integer>> m_nj; // frequency of each cluster.
	
	// Parameters of the prior for the intercept and coefficients.
	double[] m_abNuA = new double[]{0, 1};
	double[] m_abNuB = new double[]{0, 1};
	
	_thetaStar[] m_thetaStars = new _thetaStar[1000];
	 
	public MultiTaskWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_M = 5; m_kBar = 1;
		m_nj = new TreeMap<Integer, ArrayList<Integer>>();
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	// calculate the loglikelihood of the user given the 
	public double calcLogLikelihood(_AdaptStruct user, int k){
		double L = 0, Pi = 0;
		for(_Review review: user.getReviews()){
			if(review.getType() != rType.ADAPTATION)
				continue;
			Pi = logit(review.getSparse(), review.getYLabel(), user.getId());
			L += Math.log(Pi);
		}
		if(m_LNormFlag)
			return L/getAdaptationSize(user);
		else
			return L;
	}
	
	public double logit(_SparseFeature[] fvs, int yLabel, int index){
		int cIndex = m_J[index];
		double[][] beta = m_thetaStars[cIndex].m_beta;
		double[] prob = dotProduct(beta, fvs);
		if(yLabel == 1)
			return prob[1]/Utils.sumOfArray(prob);
		else
			return prob[0]/Utils.sumOfArray(prob);
	}
	public double[] dotProduct(double[][] beta, _SparseFeature[] fvs){
		int classNo = beta[0].length;
		double[] prob = new double[classNo]; //prob of x*beta for different classes.
		for(int i=0; i<classNo; i++){
			prob[i] = beta[i][0];//bias term.
			for(_SparseFeature fv: fvs)
				prob[i] += beta[i][fv.getIndex()+1]*fv.getValue();
		}
		return prob;
	}
	// The main mcmc algorithm, assign each 
	public void calculate_E_step(){
		double[] prob;
		double denomiator, u;
		int cIndex, picked;
		_thetaStar phi;
		for(int i=0; i<m_userList.size(); i++){
			_LinAdaptStruct user = (_LinAdaptStruct) m_userList.get(i);
			cIndex = m_J[user.getId()];
			m_nj.get(cIndex).add(i);
			if(m_nj.get(cIndex).size() == 0){// No data associated with the cluster.
				phi = m_thetaStars[cIndex];
				m_nj.remove(cIndex);
				// Shift m_thetastars.
				for(int j=cIndex; j<m_thetaStars.length-1; j++)
					m_thetaStars[j] = m_thetaStars[j+1];
				// Shift user cluster assignment.
				for(int k=0; k<m_J.length; k++){
					if(m_J[k] > cIndex)
						m_J[k]--;
				}
				m_kBar = m_nj.size();
				m_thetaStars[m_kBar+1] = phi;
				sampleThetaStars(m_kBar+1, m_M-1);
			} else{
				m_kBar = m_nj.size();
				sampleThetaStars(m_kBar, m_M);
			}
			// Calculate the probability of each cluster.
			prob = new double[m_kBar+m_M];
			denomiator =  Math.log(m_userList.size() - 1 + m_alpha);
			for(int k=0; k<m_kBar; k++){// existing cluster
				prob[k] = calcLogLikelihood(user, k);
				prob[k] += Math.log(m_nj.get(k).size()) - denomiator;
			}
			for(int m=0; m<m_M; m++){// new cluster
				prob[m+m_kBar] = calcLogLikelihood(user, m_kBar+m);
				prob[m+m_kBar] += Math.log(m_alpha) - Math.log(m_M) - denomiator;
			}
			normalizeProb(prob);
				
			// Pick one cluster assignment for the current instance.
			u = Math.random();
			picked = findIndex(prob, u);
				
			if(picked <= m_kBar){
				m_J[i] = picked;
				m_nj.get(picked).add(i);
				dltThetaStars(m_kBar+1); // delete the auxiliary parameters.
			} else{
				m_J[i] = m_kBar+1;
				m_nj.put(m_kBar+1, new ArrayList<Integer>(i));
				// Assign the picked cluster to the first new cluster.
				m_thetaStars[m_kBar+1] = m_thetaStars[picked];
				dltThetaStars(m_kBar+2);
			}
		}
	}
	// Sample the weights given the cluster assignment.
	public void calculate_M_step(){
		double[] relatedBeta;
		_thetaStar curThetaStar;
		ArrayList<Integer> members;
		// Use instances inside one cluster to update the thetastar.
		for(int cIndex: m_nj.keySet()){
			curThetaStar = m_thetaStars[cIndex];
			members = m_nj.get(cIndex);
			for(int m: members){
				m_thetaStars[cIndex].m_nuA = Math.sqrt(sampleSigBeta(curThetaStar.getBias(), curThetaStar.m_nuA*curThetaStar.m_nuA, m_abNuA));
				m_thetaStars[cIndex].m_nuB = Math.sqrt(sampleSigBeta());
				m_thetaStars[cIndex].scaleSigComp();
				m_thetaStars[cIndex].m_beta = sampleBeta(m, curThetaStar.m_beta, curThetaStar.m_sigComp);	
			}
		}
	}	
	
	public double calcLogPostSigma(double[] beta, double x, double[] param){
		double sigma = Math.sqrt(Math.log(x));
		double logPost = -beta.length*(Math.log(Math.sqrt(2*Math.PI*sigma*sigma)))-sumSquare(beta)/(2*sigma*sigma)
						 -Math.log(Math.sqrt(Math.PI*2*param[0]*param[0]))-(x-param[0])*(x-param[0])/(2*param[1]*param[1]);
		return logPost;
	}
	public double calcLogPostBeta(int uIndex, double[][] beta, double[][] sigma){
		double logLike = calcLogLikelihood(m_userList.get(uIndex), m_J[uIndex]);
		double logPrior = calcPrior(beta, sigma);
		return logLike + logPrior;
	}
	// beta ~ N(0, sigma)
	public double calcPrior(double[][] beta, double[][] sigma){
		double val = 0;
		for(int i=0;i < beta.length; i++){
			for(int j=0; j<beta[0].length; j++){
				val += -0.5*beta[i][j]*beta[i][j]/(sigma[i][j]*sigma[i][j]);
			}
		}
		return val;
	}

	// Accumulate the sum the elements of the array.
	public void cumsum(double[] arr){
		double sum = 0;
		for(int i=0; i<arr.length; i++){
			sum += arr[i];
			arr[i] = sum;
		}
	}
	// Delete the element in thetaStar from 'start'.
	public void dltThetaStars(int start) {
		for (int i = start; i < m_kBar+m_M; i++) {
			m_thetaStars[i] = null;
		}
	}
	public void DPMNL(){
		for(int i=0; i<m_numberOfIterations; i++){
			calculate_E_step();
			calculate_M_step();
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
	public void loadUsers(ArrayList<_User> userList) {
		super.loadUsers(userList);
		m_J = new int[userList.size()];// Cluster assignment.
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
			m_thetaStars[start+m].m_nuA = sqrtExpNormrndOne(m_abNuA[0], m_abNuA[1]);
			m_thetaStars[start+m].m_nuB = sqrtExpNormrndOne(m_abNuB[0], m_abNuB[1]);
			m_thetaStars[start+m].scaleSigComp();
			m_thetaStars[start+m].setBeta(m_normal);
		}
	}
	public double sampleSigBeta(double[] beta, double sigma2, double[] param){
		double x = Math.log(sigma2);
		double z = calcLogPostSigma(beta, x, param);
		return Math.exp(z);
	}
	// Sample the cluster weights with fixed cluster assignment.
	public double[][] sampleBeta(int uIndex, double[][] beta, double[][] sigma){
		double postBeta = calcLogPostBeta(uIndex, beta, sigma);
		
	}
	public double sumSquare(double[] beta){
		double val = 0;
		for(double b: beta)
			val += b*b;
		return val;
	}
	@Override
	protected void setPersonalizedModel() {
		// TODO Auto-generated method stub
		
	}
	
	// Sqrt+Exp+Normalize for an array from normal distributions parameterized by values in the two parameters.
	public double[] sqrtExpNormrnd(double[] us, double[] sigmas){
		if(us.length == 0 || sigmas.length == 0 || us.length != sigmas.length)
			return null;
		double[] rnds = new double[us.length];
		for(int i=0; i<us.length; i++){
			rnds[i] = sqrtExpNormrndOne(us[i], sigmas[i]);
		}
		return rnds;
	}
	// Sqrt+Exp+Normalize for one random value.
	public double sqrtExpNormrndOne(double u, double sigma){
		return Math.sqrt(Math.exp(m_normal.nextDouble(u, sigma)));
	}
	
	public int[] unique(int[] arr){
		HashSet<Integer> set = new HashSet<Integer>();
		for(int i: arr)
			set.add(i);
		ArrayList<Integer> list = new ArrayList<Integer>(set);
		Collections.sort(list);
		int[] res = new int[list.size()];
		for(int i=0; i<list.size(); i++)
			res[i] = list.get(i);
		return res;
	} 
	
	public static void main(String[] args){
		double[] p = new double[]{0.1, 0.3, 0.5, 0.75, 0.88, 1};
		MultiTaskWithDP test = new MultiTaskWithDP(0, 0, null, "", "");
		System.out.println(test.findIndex(p, 1));	
	}
}
