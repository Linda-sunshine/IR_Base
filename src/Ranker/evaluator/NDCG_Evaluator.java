/**
 * 
 */
package Ranker.evaluator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import structures._QUPair;
import structures._Query;

/**
 * @author wang296
 * Evaluate NDCG@k (at most 50)
 */
public class NDCG_Evaluator extends Evaluator {
	int m_k;
	double[] m_discount,m_gain;
	double m_iDCG;
	
	public NDCG_Evaluator(int k) {
		super();
		m_k = Math.min(50, k);
		
		m_discount = new double[50];
		for(int i=0; i<50; i++)
			m_discount[i] = 1.0/Math.log(i+2);
		
		m_gain = new double[5];
		for(int i=0; i<5; i++)
			m_gain[i] = Math.pow(2.0, i)-1;
	}
	
	@Override
	public void setQuery(_Query query) {//every time we want to evaluate new queries
		super.setQuery(query);
		
		ArrayList<Integer> labels = new ArrayList<Integer>(m_size);
		for(_QUPair qu:m_query.m_docList)
			labels.add(qu.m_y);
		Collections.sort(labels, Collections.reverseOrder());
		
		//calculate iDCG
		m_iDCG = 0;		
		for(int i=0; i<Math.min(labels.size(),m_k); i++)
			m_iDCG += m_gain[labels.get(i)] * m_discount[i];
		
		updateDeltas();
	}
	
	@Override
	public void updateDeltas(){
		super.updateDeltas();
		
		//create cache for delta
		HashMap<_QUPair, Double> change;
		_QUPair qu1, qu2;
		double delta;
		for(int i=0; i<Math.min(m_size, m_k); i++){
			qu1 = m_query.m_docList.get(i);
			change = new HashMap<_QUPair, Double>();			
			for(int j=i+1; j<m_size; j++){
				qu2 = m_query.m_docList.get(j);
				if (j>=m_k){
					delta = (m_gain[qu1.m_y]-m_gain[qu2.m_y]) * m_discount[i];
				} else {
					delta = (m_gain[qu1.m_y]-m_gain[qu2.m_y]) * (m_discount[i]-m_discount[j]);
				}
				change.put(qu2, Math.abs(delta)/m_iDCG);
			}
			m_deltas.put(qu1, change);
		}
	}
	
	@Override
	public double eval(_Query query) {
		setQuery(query);
		
		double DCG = 0;
		for(int i=0; i<Math.min(m_query.m_docList.size(),m_k); i++)
			DCG += m_gain[m_query.m_docList.get(i).m_y] * m_discount[i]; 
		return DCG/m_iDCG;
	}
}
