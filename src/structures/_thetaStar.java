package structures;

import cern.jet.random.tdouble.Normal;

public class _thetaStar{
	public int m_index;
	public int m_dim;
	public int m_memSize;
	public double m_u; // mean of the distribution
	public double m_sigma; // Sigma for the distribution.
	public double[] m_beta;
	
	public _thetaStar(int dim){
		m_dim = dim;
		m_memSize = 0;
		m_u = 0;
		m_sigma = 1;
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
	public void setSigma(double s){
		m_sigma = s;
	}
	public void setIndex(int i){
		m_index = i;
	}
	public int getIndex(){
		return m_index;
	}
	public void setBeta(Normal normal){
		for(int i=0; i<m_dim; i++)
			m_beta[i] = normal.nextDouble(m_u, m_sigma);
	}
}