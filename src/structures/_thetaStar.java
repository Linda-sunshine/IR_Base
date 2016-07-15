package structures;

import java.util.Arrays;

import cern.jet.random.tdouble.Normal;

public class _thetaStar{
	public int m_dim;
	public int m_memSize;
	public double m_nuA;
	public double m_nuB;
	public double[] m_sigComp;
	public double[] m_sigBeta;
	public double[] m_beta;
	
	public _thetaStar(int dim){
		m_dim = dim;
		m_memSize = 0;
		m_nuA = 1;
		m_nuB = 1;
		init();
	}
	public void init(){
		m_sigComp = new double[m_dim];
		m_sigBeta = new double[m_dim];
		Arrays.fill(m_sigBeta, 1);
		m_beta = new double[m_dim];
	}
	public int getMemSize(){
		return m_memSize;
	}
	public void memSizeMinusOne(){
		m_memSize--;
	}
	public void memSizeAddOne(){
		m_memSize++;
	}
	public double[] normrnd(double u, double[] sigmas, Normal normal){
		double[] rnds = new double[sigmas.length];
		for(int i=0; i<sigmas.length; i++)
			rnds[i] = normal.nextDouble(u, sigmas[i]);
		return rnds;
	}
	public void scaleSigComp(){
		m_sigComp[0] = m_nuA*m_sigBeta[0];
		for(int i=1; i<m_sigComp.length; i++)
			m_sigComp[i] = m_nuB*m_sigBeta[i];
	}
	public void setBeta(Normal normal){
		scaleSigComp();
		m_beta = normrnd(0, m_sigComp, normal);
	}
}