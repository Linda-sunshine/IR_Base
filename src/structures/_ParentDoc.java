package structures;

import java.util.ArrayList;
import java.util.HashMap;

import utils.Utils;

public class _ParentDoc extends _Doc {

	public ArrayList<_ChildDoc> m_childDocs;
	HashMap<Integer, Integer> m_word2Index;

	public _ParentDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);

		m_childDocs = new ArrayList<_ChildDoc>();

		setName(name);
		setTitle(title);
	}
	
	public void addChildDoc(_ChildDoc cDoc){
		m_childDocs.add(cDoc);
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha) {
		super.setTopics4Gibbs(k, alpha);
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	
	public void collectTopicWordStat() {
		for (_Word w:m_words) {
			m_phi[m_word2Index.get(w.getIndex())][w.getTopic()]++;
		}
	}
	
	public void estStnTheta() {
		double[] theta;
		for(_Stn s:m_sentences) {
			theta = s.m_topics;
			for(_SparseFeature f:s.m_x_sparse) {
				for(int tid=0; tid<m_topics.length; tid++)
					theta[tid] += m_phi[m_word2Index.get(f.m_index)][tid]; 
			}
			Utils.L1Normalization(theta);
		}
	}
}
