package structures;

import java.util.Arrays;
import java.util.HashMap;

public class _ParentDoc4DCM extends _ParentDoc{
	
	public double[][] m_wordTopic_stat;
	public double[][] m_wordTopic_prob;
	
	public _ParentDoc4DCM(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		
		
	}
	
	protected void setWordTopicStat(int k, int vocalSize){
		m_wordTopic_stat = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_stat[i], 0);
		
		m_wordTopic_prob = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_prob[i], 0);
	}
	
	public void setTopics4Gibbs(int k, double alpha, int vocalSize) {
		createSpace(k, alpha);
		setWordTopicStat(k, vocalSize);
		
		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse) {
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++) {
				tid = m_rand.nextInt(k);
				m_words[wIndex] = new _Word(wid, tid);// randomly initializing the topics inside a document
				m_sstat[tid] ++; // collect the topic proportion
				
				m_wordTopic_stat[tid][wid] ++;
				wIndex ++;
			}
		}
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	

}
