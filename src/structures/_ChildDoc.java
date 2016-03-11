package structures;

import utils.Utils;

public class _ChildDoc extends _Doc {
	public int[][] m_xTopicSstat;//joint assignment of <x,z>: 0 from general, 1 from specific
	public int[] m_xSstat; // sufficient statistics for x

	public double[][] m_xTopics; // proportion of topics (0 from general, 1 from specific)
	public double[] m_xProportion; // proportion of x

	public _ParentDoc m_parentDoc;
	
	public _ChildDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);
		m_parentDoc = null;
		m_name = name;
		m_title = title;
	}
	
	public void setParentDoc(_ParentDoc pDoc){
		m_parentDoc = pDoc;
	}
	
	public void createXSpace(int k, int gammaSize) {
		m_xTopicSstat = new int[gammaSize][k];
		m_xTopics = new double[gammaSize][k];
		m_xSstat = new int[gammaSize];
		m_xProportion = new double[gammaSize];
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){		
		createSpace(k, alpha);
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				tid = m_rand.nextInt(k);
				xid = m_rand.nextInt(gammaSize);
				m_words[wIndex] = new _Word(wid, tid, xid);

				m_xTopicSstat[xid][tid] ++;
				m_xSstat[xid] ++;
						
				wIndex ++;
			}
		}
	}
	
	public void estGlobalLocalTheta() {
		Utils.L1Normalization(m_xProportion);
		for(int x=0; x<m_xTopics.length; x++)
			Utils.L1Normalization(m_xTopics[x]);
		
		for(_Word w: m_words){
			Utils.L1Normalization(w.m_xProb);
		}
	}
}
