package structures;

public class _ChildDoc4LogisticRegression extends _ChildDoc{

	public _ChildDoc4LogisticRegression(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		int wid, tid, xid, wIndex = 0, localIndex = 0, gammaSize=m_xSstat.length;
		
		for(_SparseFeature fv : m_x_sparse){
			wid = fv.getIndex();
			
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				xid = m_rand.nextInt(gammaSize);
				
				m_words[wIndex] = new _Word(wid, tid, xid, localIndex, fv.getValues());
			
				xid = m_words[wIndex].getX();
				m_xTopicSstat[xid][tid] ++;
				m_xSstat[xid] ++;
				
				wIndex ++;
			}
			localIndex ++;
		}
	}
}