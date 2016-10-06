package structures;

public class _Clique implements Comparable<_Clique> {
	int m_dimension;//3_clique or 4_clique or k_clique.
	int[] m_indexes;
	double m_value;

	public _Clique(){
		m_dimension = 3;
		m_indexes = new int[m_dimension];
		m_value = 0;
	}
	public _Clique(int[] indexes, double v){
		m_dimension = indexes.length;
		m_indexes = indexes;
		m_value = v;
	}
	
	public int getDimension(){
		return m_dimension;
	}
	public int[] getIndexes(){
		return m_indexes;
	}
	
	public int compareTo(_Clique it) {
		if (this.m_value < it.m_value)
			return -1;
		else if (this.m_value > it.m_value)
			return 1;
		else
			return 0;
	}
	public double getValue(){
		return m_value;
	}
}
