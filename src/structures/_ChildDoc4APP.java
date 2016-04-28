package structures;

import java.util.HashMap;

import utils.Utils;

public class _ChildDoc4APP extends _ChildDoc{
	public HashMap<Integer, Integer> m_wordXStat;
	
	public _ChildDoc4APP(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		
	}
	
	public void createXSpace(int k_description, int k_review, int gammaSize){
		m_xTopicSstat = new double[gammaSize][];
		m_xTopics = new double[gammaSize][];
		
		m_xTopicSstat[0] = new double[k_description];
		m_xTopicSstat[1] = new double[k_review];
		m_xTopics[0] = new double[k_description];
		m_xTopics[1] = new double[k_review];
				
		m_xSstat = new double[gammaSize];
	
		m_xProportion = new double[gammaSize];
	}
	
	public void setTopics4Gibbs(int k_description, int k_review, double alpha){
		createSpace(k_description+k_review, alpha);
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(gammaSize);
				
				if(xid==0){
					tid = m_rand.nextInt(k_description);
					m_words[wIndex] = new _Word(wid, tid, xid);
					m_xTopicSstat[xid][tid] ++;
					m_xSstat[xid] ++;
					m_wordXStat.put(wid, m_wordXStat.get(wid)+1);
				}else{
					tid = m_rand.nextInt(k_review); 
					m_words[wIndex] = new _Word(wid, tid, xid);
					
					m_xTopicSstat[xid][tid] ++;
					m_xSstat[xid] ++;
				}
				
				wIndex ++;			
			}	
		}
		
		
	}
	
}
