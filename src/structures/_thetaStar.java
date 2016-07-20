package structures;

import cern.jet.random.tdouble.Normal;

public class _thetaStar {
	int m_index;
	int m_dim;
	int m_memSize;

	double[] m_abNuA;
	double[] m_beta;
	double m_proportion;
	
	public _thetaStar(int dim, double[] abNuA){
		m_dim = dim;
		m_memSize = 0;
		m_beta = new double[m_dim];
		m_abNuA = abNuA;
	}
	
	public int getMemSize(){
		return m_memSize;
	}
	
	public void updateMemCount(int c){
		m_memSize += c;
	}
	
	public void setProportion(double p) {
		m_proportion = p;
	}
	
	public double getProportion() {
		return m_proportion;
	}
	
	public void setIndex(int i){
		m_index = i;
	}
	
	public int getIndex(){
		return m_index;
	}

	public void sampleBeta(Normal normal){
		for(int i=0; i<m_dim; i++)
			m_beta[i] = normal.nextDouble(m_abNuA[0], m_abNuA[0]);
	}
	
	public double[] getModel() {
		return m_beta;
	}
}