package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import utils.Utils;

public class _ParentDoc4ThreePhi extends _ParentDoc{
	
	//used to record all word assigned x=1 including parent and child
	public double m_pairWord;
	public double[] m_pairWordTopicSstat;
	public double[] m_pairWordTopicProb;

	//x=0
	public double m_globalWord;
	//x=1 
	public double m_parentWord;
	
	
	public double m_parentProb;
	
	public int m_xSize;
	
	public _ParentDoc4ThreePhi(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);


	}
	
	public void createXSpace(int k, int gammaSize){
		m_xSize = gammaSize;
	}
	
	public void createLocalWordTopicDistribution(int vocalbularySize, double beta){
		beta *= 0.01;
		m_pairWordTopicSstat = new double[vocalbularySize];
		m_pairWordTopicProb = new double[vocalbularySize];
		
		m_globalWord = 0;
		m_pairWord = beta*vocalbularySize;
		m_parentWord = 0;
		m_parentProb = 0;
		
		Arrays.fill(m_pairWordTopicSstat, beta);
		Arrays.fill(m_pairWordTopicProb, 0);
		
	}
	
	public void setTopics4Gibbs(int k, double alpha) {
		createSpace(k, alpha);
		
		m_topics = new double[k+1];
		m_sstat = new double[k+1];
		
		//m_sstat[k]==m_parentWord;
		for(int i=0; i<k; i++)
			m_sstat[i] = alpha;
		
		int wIndex = 0, wid, tid, xid;
		
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(m_xSize);
				tid = 0;
				if(xid==0){
					tid = m_rand.nextInt(k);
					m_globalWord ++;
					m_sstat[tid] ++;
				}else if(xid==1){
					tid = k;
					m_parentWord ++;
					m_sstat[tid] ++;
				}
				
				m_words[wIndex] = new _Word(wid, tid, xid);
				
				
				wIndex ++;
			}
		}
		
		m_phi = new double[m_x_sparse.length][k+1];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++){
			m_word2Index.put(m_x_sparse[i].m_index, i);
		}
	}
	
	public void estGlobalLocalTheta(){
		Utils.L1Normalization(m_topics);
		Utils.L1Normalization(m_pairWordTopicProb);
	}
	
	public void collectLocalWordSstat(){
		for(int i=0; i<m_pairWordTopicSstat.length; i++){
			m_pairWordTopicProb[i] += m_pairWordTopicSstat[i];
		}
		
		m_parentProb += m_parentWord;
	} 
	
}
