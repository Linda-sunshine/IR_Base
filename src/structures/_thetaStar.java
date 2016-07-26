package structures;

public class _thetaStar {
	int m_index;
	int m_dim;
	int m_memSize;

	double[] m_beta;
	double m_proportion;
	
	double m_pCount, m_nCount; // number of positive and negative documents in this cluster
	
	public _thetaStar(int dim){
		m_dim = dim;
		m_memSize = 0;
		m_beta = new double[m_dim];
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
	
	public double[] getModel() {
		return m_beta;
	}
	
	public void resetCount() {
		m_pCount = 0;
		m_nCount = 0;
	}
	
	public void incPosCount() {
		m_pCount++;
	}
	
	public void incNegCount() {
		m_nCount++;
	}
	
	public String showStat() {
		return String.format("%d(%.2f,%.1f)", m_memSize, m_pCount/(m_pCount+m_nCount), (m_pCount+m_nCount)/m_memSize);
	}
}