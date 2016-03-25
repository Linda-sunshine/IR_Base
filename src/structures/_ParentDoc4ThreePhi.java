package structures;

import java.util.Arrays;
import java.util.HashMap;

import utils.Utils;

public class _ParentDoc4ThreePhi extends _ParentDoc{
	
	public int[] m_pairWordSstat;
	public double[] m_pairWordDistribution;
	public int m_pairWord;
	
	public int[][] m_xTopicSstat;
	public int[] m_xSstat;
	
	public double[] m_xProportion;
	
	public _ParentDoc4ThreePhi(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
	}
	
	public void createXSpace(int k, int gammaSize, int vocalSize) {
		m_xTopicSstat = new int[gammaSize][];
		
		m_xTopicSstat[0] = new int[k];
		
		m_xTopicSstat[1] = new int[1];
  
		m_xSstat = new int[gammaSize];
		m_xProportion = new double[gammaSize];

		m_pairWordSstat = new int[vocalSize];
		m_pairWordDistribution = new double[vocalSize];

		Arrays.fill(m_xTopicSstat[0], 0);
		Arrays.fill(m_xTopicSstat[1], 0);

		Arrays.fill(m_xSstat, 0);
		Arrays.fill(m_xProportion, 0);

		Arrays.fill(m_pairWordSstat, 0);
		Arrays.fill(m_pairWordDistribution, 0);
	}
	
	public void setTopics4Gibbs(int k, double alpha) {
		createSpace(k + 1, alpha);
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(gammaSize);
				tid = 0;
				if(xid==0){
					tid = m_rand.nextInt(k);
					m_xTopicSstat[xid][tid]++;
					m_xSstat[xid]++;
				}else if(xid==1){
					tid = k;
					m_xTopicSstat[xid][0]++;
					m_xSstat[xid]++;
					
					m_pairWordSstat[wid]++;
					m_pairWord++;
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

		Utils.L1Normalization(m_xProportion);

		Utils.L1Normalization(m_pairWordDistribution);
	}
	
	
 }
