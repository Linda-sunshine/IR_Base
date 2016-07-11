package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.util.ArrayList;
import java.util.Arrays;

import structures._User;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DoubleMersenneTwister;
/***
 * This is a translation of dpMnl from matlab to java.
 * "Nonlinear models using dirichlet process mixtures"
 * @author lin
 */
public class MultiTaskWithDP extends ModelAdaptation{
	Normal m_normal; // Normal distribution.
	int m_leapFrog = 200; // Number of steps for the Hamiltonian dynamics.
	int m_M, m_kBar; // The number of auxiliary components.
	
	double m_eps = 0.2; // This is the constant multiplier for the step size.
	double m_a0 =-3, m_b0 = 2; // alpha~Gamma(a0, b0)
	double m_alpha = 0.001; // Scale parameter of DP.
	
	int[] m_J; // cluster identifier.
	ArrayList<Integer> m_nj = new ArrayList<Integer>(); // frequency of each cluster.
	
	double[] m_mu0, m_Sigma0, m_mu00, m_Sigma00;
	double[] m_muMu, m_SigMu;
	double m_aSigma00 = 0, m_bSigma00 = 1, m_muSig = 0, m_sigSig = 1;
	
	// Parameters of the prior for the intercept and coefficients.
	double[] m_abNuA = new double[]{0, 1};
	double[] m_abNuB = new double[]{0, 1};
	
	class _thetaStar{
		double[] m_mu;
		double[] m_sd;
		double m_nuA;
		double m_nuB;
		double[][] m_sigBeta;
		double[][] m_sigComp;
		double[][] m_beta;
		
		public _thetaStar(int dim, int classNo){
			m_mu = new double[dim];
			m_sd = new double[dim];
			m_nuA = 1;
			m_nuB = 1;
			m_sigBeta = new double[dim+1][classNo];
			m_sigComp = new double[dim+1][classNo];
			m_beta = new double[dim+1][classNo];
		}
		
		public void init(){
			Arrays.fill(m_sd, 2);
			fill2DimArray(m_sigBeta, 1);
			fill2DimArray(m_sigComp, 1);
			scaleSigComp();
		}
		public void scaleSigComp(){
			// Init sigComp row by row.
			m_sigComp[0] = scaling(m_nuA, m_sigBeta[0]);
			for(int i=1; i<m_sigComp.length; i++){
				m_sigComp[i] = scaling(m_nuB, m_sigBeta[i]);
			}
		}
		public void setBeta(){
			m_beta = normrnd(0, m_sigComp);
		}
		// Scale the array by a, arr = a*arr.
		public double[] scaling(double a, double[] arr){
			if(arr.length == 0)
				return null;
			double[] res = new double[arr.length];
			for(int i=0; i<arr.length; i++){
				res[i] = a * arr[i];
			}
			return res;
		}
		// Fill the two dimension array with val.
		public void fill2DimArray(double[][] arr, double val){
			for(int i=0; i<arr.length; i++)
				Arrays.fill(arr[i], val);
		}
	}
	
	_thetaStar[] m_thetaStars = new _thetaStar[m_kBar + m_M];
	public MultiTaskWithDP(int classNo, int featureSize) {
		super(classNo, featureSize);
		m_J = new int[m_userList.size()];
		Arrays.fill(m_J, 1);
		m_M = 5; m_kBar = 1;
		m_normal = new Normal(0, 1, new DoubleMersenneTwister());
	}

	public void dpMnl(){
		
	}
	
	// The main mcmc algorithm.
	public void MCMC(){
		for(int i=0; i<m_userList.size(); i++){
			_LinAdaptStruct user = (_LinAdaptStruct) m_userList.get(i);
			for(int m=0; m<m_M; m++){
				m_thetaStars[m+m_kBar].m_sd = sqrtExpNormrnd(m_mu00, m_Sigma00);
				m_thetaStars[m+m_kBar].m_mu = normrnd(m_mu0, m_Sigma0);
				m_thetaStars[m+m_kBar].m_nuA = sqrtExpNormrndOne(m_abNuA[0], m_abNuA[1]);
				m_thetaStars[m+m_kBar].m_nuB = sqrtExpNormrndOne(m_abNuB[0], m_abNuB[1]);
				m_thetaStars[m+m_kBar].scaleSigComp();
				m_thetaStars[m+m_kBar].setBeta();
			}
			double[] prob = new double[m_kBar+m_M];
			double denomiator =  Math.log(m_userList.size() - 1 + m_alpha);
			for(int k=0; k<m_kBar; k++){
				prob[k] = calcLogLikelihood(user, k);
				prob[k] += Math.log(m_nj.get(k)) - denomiator;
			}
			for(int m=0; m<m_M; m++){
				prob[m+m_kBar] = calcLogLikelihood(user, m_kBar+m);
				prob[m+m_kBar] += Math.log(m_alpha) - Math.log(m_M) - denomiator;
			}
		}
	}

	public double calcLogLikelihood(){
		
	}
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		// TODO Auto-generated method stub
		
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
	public double[][] normrnd(double u, double[][] sigmas){
		double[][] rnds = new double[sigmas.length][sigmas[0].length];
		for(int i=0; i<sigmas.length; i++){
			for(int j=0; j<sigmas[0].length; j++){
				rnds[i][j] = m_normal.nextDouble(u, sigmas[i][j]);
			}
		}
		return rnds;
	}

}
