package structures;

import utils.Utils;

public class _ChildDoc4BaseWithPhi_Hard extends _ChildDoc4BaseWithPhi{
	public _ChildDoc4BaseWithPhi_Hard(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);
		
		// TODO Auto-generated constructor stub
	}
	
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		_SparseFeature[] parentFv = m_parentDoc.getSparse();
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		tid = 0;
		for(_SparseFeature fv:m_x_sparse){
			
			wid = fv.getIndex();
			
			for(int j=0; j<fv.getValue(); j++){

				if(Utils.indexOf(parentFv, wid)!=-1){
					xid = 0;
					tid = m_rand.nextInt(k);
					m_xTopicSstat[xid][tid] ++;
					m_xSstat[xid] ++;
				}else{
				
					xid = m_rand.nextInt(gammaSize);
					if(xid == 0){
						tid = m_rand.nextInt(k);
						m_xTopicSstat[xid][tid]++;
						m_xSstat[xid]++;
					}else if(xid==1){
						tid = k ;
						m_xTopicSstat[xid][wid]++;
						m_xSstat[xid]++;
						m_childWordSstat ++;
					}
				}
				
				m_words[wIndex] = new _Word(wid, tid, xid);
				
				wIndex ++;
			}
		}
	}

}
