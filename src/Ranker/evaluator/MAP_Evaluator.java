/**
 * 
 */
package Ranker.evaluator;

import java.util.HashMap;

import structures._QUPair;
import structures._Query;

/**
 * @author wang296
 * Evaluate the MAP performance
 */
public class MAP_Evaluator extends Evaluator implements Evaluation {	
	double m_P;//number of relevant documents
	
	public MAP_Evaluator(){
		super();	
		m_P = 0;
	}
	
	@Override
	public void setQuery(_Query query) {
		super.setQuery(query);	
		
		m_P = 0;		
		for(_QUPair d:m_query.m_docList)
			if (d.m_y>0)
				m_P++;
		
		if (m_P>0)
			updateDeltas();
	}
	
	@Override
	public void updateDeltas(){
		super.updateDeltas();
		
		//create cache for delta
		HashMap<_QUPair, Double> change = null;
		_QUPair qu1, qu2;
		double delta;
		for(int i=0; i<m_size; i++){
			qu1 = m_query.m_docList.get(i);
			if (change == null)
				change = new HashMap<_QUPair, Double>();		
			delta = 1.0/(i+1.0);
			for(int j=i+1; j<m_size; j++){				
				qu2 = m_query.m_docList.get(j);
				if (qu1.m_y != qu2.m_y) //make a difference
					change.put(qu2, delta/m_P);//absolute diff
				delta += 1.0/(j+1.0);
			}
			if (change.size()>0){
				m_deltas.put(qu1, change);
				change = null;
			}
		}
	}
	
	@Override
	public double eval(_Query query) {
		setQuery(query);//including sort by ranking score		
		
		double ap = 0;
		m_P = 0;
		for(int i=0; i<m_size; i++){
			if (m_query.m_docList.get(i).m_y>0){
				m_P++;
				ap += m_P/(i+1);
			}
		}
		if (m_P==0)
			return -1;
		return ap/m_P;
	}
}
