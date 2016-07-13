package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

import structures._User;
import structures._thetaStar;
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
	double m_a0 =-3, m_b0 = 2; // alpha~Gamma(a0, b0)
	double m_alpha = 0.001; // Scale parameter of DP.
	
	int[] m_J; // cluster assignment of users.
	TreeMap<Integer, Integer> m_nj; // frequency of each cluster.
	
//	double[] m_mu0, m_Sigma0, m_mu00, m_Sigma00;
//	double[] m_muMu, m_SigMu;
//	double m_aSigma00 = 0, m_bSigma00 = 1, m_muSig = 0, m_sigSig = 1;
	
	// Parameters of the prior for the intercept and coefficients.
	double[] m_abNuA = new double[]{0, 1};
	double[] m_abNuB = new double[]{0, 1};
	
	_thetaStar[] m_thetaStars = new _thetaStar[m_kBar + m_M];
	
	public MultiTaskWithDP(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap){
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_M = 5; m_kBar = 1;
		m_nj = new TreeMap<Integer, Integer>();
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	//
	public double calcLogLikelihood(_AdaptStruct u, int k){
		return 0;
	}
	
	// Accumulate the sum the elements of the array.
	public void cumsum(double[] arr){
		double sum = 0;
		for(int i=0; i<arr.length; i++){
			sum += arr[i];
			arr[i] = sum;
		}
	}
	// Delete the element in thetaStar form 'start'.
	public void dltThetaStars(int start) {
		_thetaStar[] tmp = Arrays.copyOf(m_thetaStars, m_thetaStars.length);
		m_thetaStars = new _thetaStar[start];
		for (int i = 0; i < start; i++) {
			m_thetaStars[i] = tmp[i];
		}
	}
	public void dpMnl(){
		for(int i=0; i<m_numberOfIterations; i++){
			MCMC();
			remix();
			pickAlpha();
			
			if(rem(i, 5) == 0)
				m_eps = 0.4;
			else 
				m_eps = 0.5;
			
			if(i > m_burnIn)
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

	// The main mcmc algorithm.
	public void MCMC(){
		double[] prob;
		double denomiator, maxP, u;
		int cIndex, picked;
		_thetaStar phi;
		for(int i=0; i<m_userList.size(); i++){
			_LinAdaptStruct user = (_LinAdaptStruct) m_userList.get(i);
			cIndex = m_J[user.getId()];
			m_nj.put(cIndex, m_nj.get(cIndex) -1);
			if(m_nj.get(cIndex) == 0){// No data associated with the cluster.
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
				
				
			}
			
			for(int m=0; m<m_M; m++){
				m_thetaStars[m+m_kBar].m_nuA = sqrtExpNormrndOne(m_abNuA[0], m_abNuA[1]);
				m_thetaStars[m+m_kBar].m_nuB = sqrtExpNormrndOne(m_abNuB[0], m_abNuB[1]);
				m_thetaStars[m+m_kBar].scaleSigComp();
				m_thetaStars[m+m_kBar].setBeta(m_normal);
			}
			prob = new double[m_kBar+m_M];
			denomiator =  Math.log(m_userList.size() - 1 + m_alpha);
			for(int k=0; k<m_kBar; k++){
				prob[k] = calcLogLikelihood(user, k);
				prob[k] += Math.log(m_nj[k]) - denomiator;
			}
			for(int m=0; m<m_M; m++){
				prob[m+m_kBar] = calcLogLikelihood(user, m_kBar+m);
				prob[m+m_kBar] += Math.log(m_alpha) - Math.log(m_M) - denomiator;
			}
			maxP = Utils.maxOfArrayValue(prob);
			for(int j=0; j<prob.length; j++){
				prob[j] = Math.exp(prob[j]-maxP);
			}
			Utils.L1Normalization(prob);
			cumsum(prob);
			
			u = Math.random();
			ArrayList<Integer> k0 = find(prob, u);
			picked = k0.get(0);
			
			if(picked <= m_kBar){
				m_J[i] = picked;
				m_nj[picked]++;
				dltThetaStars(m_kBar+1); // delete the auxillary parameters.
			} else{
				m_J[i] = m_kBar+1;
				m_nj.add(1);
				_thetaStar phi = m_thetaStars[picked];
				dltThetaStars(m_kBar+2);
				m_thetaStars[m_kBar+1] = phi;
			}
			
		}
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		super.loadUsers(userList);
		m_J = new int[userList.size()];// Cluster assignment.
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
	
	public void pickAlpha(){
		
	}
	
	public void remix(){
		double e;
		double[] relatedBeta;
		_thetaStar tmp;
		int[] unique = unique(m_J);
		for(int i=0; i<unique.length; i++){
			tmp = m_thetaStars[i];
//			X
//			Y 
//			nY
			for(int j=0; j<m_featureSize; j++){
				tmp.m_mu[j] = getMu0(X, tmp.m_mu[j], m_mu0[j], m_Sigma0[j], tmp.m_sd[j]);
				tmp.m_sd[j] = Math.sqrt(getSig0(X, tmp.m_sd[j]*tmp.m_sd[j], m_mu00[j], tmp.m_mu[j]));
			}
			relatedBeta = tmp.m_beta[0];
			tmp.m_nuA = sqrt(getSigBeta(relatedBeta, tmp.m_nuA*tmp.m_nuA, m_abNuA[0], m_abNuA[1]));
			relatedBeta = tmp.getBeta(1);
			for(int j=0;j<50;j++)
				tmp.m_nuB = Math.sqrt(getSigBeta(relatedBeta, tmp.m_nuB*tmp.m_nuB, m_abNuB[0], m_abNuB[1]));
			
			tmp.scaleSigComp();
			e = m_eps*(1/Math.sqrt(tmp.m_sigComp^2 + nY/4));
			getBeta();
			m_thetaStars[i] = tmp;	
		}
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
	


}
