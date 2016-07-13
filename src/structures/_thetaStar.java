package structures;

import java.util.Arrays;

import cern.jet.random.tdouble.Normal;

public class _thetaStar{
//	public double[] m_mu;
//	public double[] m_sd;
	public double m_nuA;
	public double m_nuB;
	double[][] m_sigBeta;
	public double[][] m_sigComp;
	public double[][] m_beta;
	
	public _thetaStar(int dim, int classNo){
//		m_mu = new double[dim];
//		m_sd = new double[dim];
		m_nuA = 1;
		m_nuB = 1;
		m_sigBeta = new double[dim+1][classNo];
		m_sigComp = new double[dim+1][classNo];
		m_beta = new double[dim+1][classNo];
	}
	
	public void init(){
//		Arrays.fill(m_sd, 2);
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
	public void setBeta(Normal normal){
		m_beta = normrnd(0, m_sigComp, normal);
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
	
	public double[][] normrnd(double u, double[][] sigmas, Normal normal){
		double[][] rnds = new double[sigmas.length][sigmas[0].length];
		for(int i=0; i<sigmas.length; i++){
			for(int j=0; j<sigmas[0].length; j++){
				rnds[i][j] = normal.nextDouble(u, sigmas[i][j]);
			}
		}
		return rnds;
	}
	// The first row of the beta is bias.
	public double[] getBias(){
		return m_beta[0];
	}
	public double[] getWeights(){
		int dim = m_beta.length-1;
		int classNo = m_beta[0].length;
		double[] weights = new double[dim*classNo];
		for(int i=0; i<dim; i++){
			for(int j=0; j<classNo; j++)
				weights[i*classNo+j] = m_beta[i+1][j];
		}
		return weights;
	}
}