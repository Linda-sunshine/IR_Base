package structures;

import cern.jet.random.tdouble.Normal;

public class _thetaStar{
	public int m_index;
	public int m_dim;
	public int m_memSize;
	public double m_prob; // the probability if the current instance belongs to the thetastar.
	public boolean m_diffFlag; // the flag decides whether we use the same distribution or not.
	public double m_u; // mean of the distribution
	public double m_sigma; // Sigma for the distribution.
	
	// In case the parameters of thetastar are sampled from different distributions.
	public double m_u2;
	public double m_sigma2;
	public double[] m_beta;
	
	// Constructor of one set of parameters.
	public _thetaStar(int dim, double[] d1){
		m_dim = dim;
		m_memSize = 0;
		m_prob = 0;
		m_u = d1[0]; m_sigma = d1[1];
		m_beta = new double[m_dim];
		m_diffFlag = false;
	}
	// Constructor of two sets of parameters.
	public _thetaStar(int dim, double[] d1, double[] d2){
		m_dim = dim;
		m_memSize = 0;
		m_prob = 0;
		m_u = d1[0]; m_sigma = d1[1];
		m_u2 = d2[0]; m_sigma2 = d2[1];
		m_beta = new double[m_dim];
		m_diffFlag = true;
	}

	public void clear(){
		m_index = 0;
		m_memSize = 0;
		m_prob = 0;
	}
	public int getMemSize(){
		return m_memSize;
	}
	public double getProb(){
		return m_prob;
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
	public void setProb(double p){
		m_prob = p;
	}
	public int getIndex(){
		return m_index;
	}
	public void setBeta(Normal normal){
		if(m_diffFlag){
			for(int i=0; i<m_dim/2; i++)
				m_beta[i] = normal.nextDouble(m_u, m_sigma);
			for(int i=m_dim/2; i<m_dim; i++)
				m_beta[i] = normal.nextDouble(m_u2, m_sigma2);
		}
		else{
			for(int i=0; i<m_dim; i++)
				m_beta[i] = normal.nextDouble(m_u, m_sigma);
		}
		
	}
	
}