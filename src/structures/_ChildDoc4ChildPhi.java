package structures;

import java.util.Arrays;

import utils.Utils;

public class _ChildDoc4ChildPhi extends _ChildDoc{
	public int m_xSize;
	
	//child-specific word topic sufficient statistic
	public double[] m_localWordTopicSstat;
	
	//child-specific word topic distribution
	public double[] m_localWordTopicProb;
	
	//statistic how many words assigned to local 
	// m_localTopicSstat = vocabularySize*beta
	public double m_localTopicSstat;
	
	//x=0
	public double m_globalWord;
	
	//x=1
	public double m_localWord;
	
	public _ChildDoc4ChildPhi(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);
		// TODO Auto-generated constructor stub
	}

	public void createXSpace(int k, int gammaSize){
		m_xSize = gammaSize;
	}
	
	public void createLocalWordTopicDistribution(int vocalbularySize, double beta){
		beta *= 0.01;
		m_localWordTopicSstat = new double[vocalbularySize];
		m_localWordTopicProb = new double[vocalbularySize];
		m_localTopicSstat = beta*vocalbularySize;
		
		Arrays.fill(m_localWordTopicSstat, beta);
		Arrays.fill(m_localWordTopicProb, 0);
		
		m_globalWord = 0;
		m_localWord = 0;
		
	}
	
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		m_topics = new double[k+1];
		m_sstat = new double[k+1];
		
		//m_sstat[k] == m_parentWord;
		//alpha is 0
		for(int i=0; i<k; i++)
			m_sstat[i] = alpha;
		
		int wIndex = 0, wid, tid, xid;
		tid = 0;
		for(_SparseFeature fv:m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(m_xSize);
				if(xid == 0){
					tid = m_rand.nextInt(k);
					m_globalWord ++;
					m_sstat[tid] ++;
				}else if(xid==1){
					tid = k;
					m_sstat[tid] ++;
					m_localWord ++;
					m_localTopicSstat ++;
					m_localWordTopicSstat[wid] ++;
				}
				
				m_words[wIndex] = new _Word(wid, tid, xid);
				
				wIndex ++;
			}
		}
	}
	
	public void estGlobalLocalTheta(){
		Utils.L1Normalization(m_topics);
		Utils.L1Normalization(m_localWordTopicProb);
	}
	
	public void collectLocalWordSstat(){
		for(int i=0; i<m_localWordTopicSstat.length; i++){
			m_localWordTopicProb[i] += m_localWordTopicSstat[i];
		}
	}

}
