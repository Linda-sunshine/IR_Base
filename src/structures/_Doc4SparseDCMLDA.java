package structures;

import java.util.Arrays;

public class _Doc4SparseDCMLDA extends _Doc4DCMLDA{
	public boolean[] m_topicIndicator;
	public double[] m_topicIndicator_prob;
	public double m_MStepIter; 
	public double m_indicatorTrue_stat;
	public double m_alphaDoc;
	public double m_topicIndicator_distribution;
	public int m_clusterIndicator;
		
	public _Doc4SparseDCMLDA(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		m_MStepIter = 0;
	}
	
	protected void setWordTopicStat(int k, int vocalSize){
		m_MStepIter = 0;
		m_alphaDoc = 0;
		m_wordTopic_stat = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_stat[i], 0);
		
		m_wordTopic_prob = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_prob[i], 0);
		
		m_topicIndicator = new boolean[k];
		for(int i=0; i<k; i++)
			m_topicIndicator[i] = false;
		
		m_topicIndicator_prob = new double[k];
		Arrays.fill(m_topicIndicator_prob, 0);
		
		m_indicatorTrue_stat = 0;
		m_topicIndicator_distribution = 0;
	}
	
	public void setTopics4Gibbs(int k, double[] alpha, int vocalSize){
		
		createSpace(k, 0);
		setWordTopicStat(k, vocalSize);
		
		boolean xid = false;
		m_alphaDoc = 0;
		for (int i = 0; i < k; i++) {
			xid = m_rand.nextBoolean();
			m_topicIndicator[i] = xid;
			if (xid == true) {
				m_indicatorTrue_stat++;
				m_alphaDoc += alpha[i];
			}

		}

		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				do {
					tid = m_rand.nextInt(k);
				} while (m_topicIndicator[tid] == false);
				m_words[wIndex] = new _Word(wid, tid);
				m_sstat[tid] ++;
				
				m_wordTopic_stat[tid][wid] ++;
				wIndex ++;
			}
		}
	}
	
	public void setWordTopicStat(int k){
		m_MStepIter = 0;
		m_alphaDoc = 0;
	
		m_topicIndicator = new boolean[k];
		for(int i=0; i<k; i++)
			m_topicIndicator[i] = false;
		
		m_topicIndicator_prob = new double[k];
		Arrays.fill(m_topicIndicator_prob, 0);
		
		m_indicatorTrue_stat = 0;
		m_topicIndicator_distribution = 0;
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){
		
		createSpace(k, 0);
		setWordTopicStat(k);
		
		boolean xid = false;
		m_alphaDoc = 0;
		for (int i = 0; i < k; i++) {
			xid = m_rand.nextBoolean();
			m_topicIndicator[i] = xid;
			if (xid == true) {
				m_indicatorTrue_stat++;
				m_alphaDoc += alpha;
			}

		}

		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				do {
					tid = m_rand.nextInt(k);
				} while (m_topicIndicator[tid] == false);
				m_words[wIndex] = new _Word(wid, tid);
				m_sstat[tid] ++;
				
				wIndex ++;
			}
		}
	}
	
	public void setTopics4GibbsCluster(int k, double[] alpha, int clusterNum, int vocalSize){
		createSpace(k, 0);
		setWordTopicStatCluster(k, vocalSize);
		
		boolean xid = false;
		m_alphaDoc = 0;
		for (int i = 0; i < k; i++) {
			xid = m_rand.nextBoolean();
			m_topicIndicator[i] = xid;
			if (xid == true) {
				m_indicatorTrue_stat++;
				m_alphaDoc += alpha[i];
			}

		}
		
		m_clusterIndicator = m_rand.nextInt(clusterNum); 

		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				do {
					tid = m_rand.nextInt(k);
				} while (m_topicIndicator[tid] == false);
				m_words[wIndex] = new _Word(wid, tid);
				m_sstat[tid] ++;
				m_wordTopic_stat[tid][wid] ++;

				wIndex ++;
			}
		}
	}
	
	protected void setWordTopicStatCluster(int k, int vocalSize){
		m_MStepIter = 0;
		m_alphaDoc = 0;
		m_wordTopic_stat = new double[k][vocalSize];
		for(int i=0; i<k; i++)
			Arrays.fill(m_wordTopic_stat[i], 0);
		
		m_topicIndicator = new boolean[k];
		for(int i=0; i<k; i++)
			m_topicIndicator[i] = false;
		
		m_topicIndicator_prob = new double[k];
		Arrays.fill(m_topicIndicator_prob, 0);
		
		m_indicatorTrue_stat = 0;
		m_topicIndicator_distribution = 0;
	}
	
}
