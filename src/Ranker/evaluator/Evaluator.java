package Ranker.evaluator;

import java.util.HashMap;

import structures._QUPair;
import structures._Query;

public class Evaluator implements Evaluation {
	_Query m_query;
	int m_size, m_i, m_j;
	HashMap<_QUPair,HashMap<_QUPair, Double>> m_deltas;
	double m_rate;
	
	public Evaluator(){
		m_query = null;
		m_size = -1;
		m_i = 0;
		m_j = 0;
		m_deltas = null;
		m_rate = 0;
	}
	
	public void setRate(double rate){
		m_rate = rate;
	}
	
	protected void sort(){
		m_query.sortDocs();
	}	
	
	@Override
	public void setQuery(_Query query) {
		m_query = query;
		m_size = query.getDocSize();
		sort();		
	}

	@Override
	public double eval(_Query query) {
		return 1.0;
	}

	@Override
	public double delta(_QUPair qu1, _QUPair qu2) {	
		if (m_deltas==null)
			return 1.0;
		else if (m_deltas.isEmpty())//empty might not be a problem (no pair swapping would affect the ranking)
			return m_rate;
		
		HashMap<_QUPair, Double> delta;
		if (m_deltas.containsKey(qu1)){
			delta = m_deltas.get(qu1);
			if (delta.containsKey(qu2))
				return m_rate+delta.get(qu2);
			else
				return m_rate;
		} else if (m_deltas.containsKey(qu2)){
			delta = m_deltas.get(qu2);
			if (delta.containsKey(qu1))
				return m_rate+delta.get(qu1);
			else
				return m_rate;
		} else
			return m_rate;		
	}

	@Override
	public void updateDeltas() {
		if (m_deltas==null)
			m_deltas = new HashMap<_QUPair,HashMap<_QUPair, Double>>();
		else
			m_deltas.clear();
	}
}