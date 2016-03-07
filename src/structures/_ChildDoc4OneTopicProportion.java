package structures;

import java.util.Arrays;

import utils.Utils;

//added by Renqin 
//Used to parent child topic model
//when x=1, only topic 0 is useful
public class _ChildDoc4OneTopicProportion extends _ChildDoc{

	//the avg probability of multiple samples.
	public double[] m_localWordProb;
	//document-specific phi
	public double[] m_localWordSstat;
	public double m_localWord;
	
	public _ChildDoc4OneTopicProportion(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);
		// TODO Auto-generated constructor stub
	}
	
	//
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		int wIndex = 0, wid, tid, xid, gammaSize = m_xSstat.length;
		for(_SparseFeature fv: m_x_sparse){
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++){
				xid = m_rand.nextInt(gammaSize);
				
				if(xid == 0)
					tid = m_rand.nextInt(k);
				else 
					tid = 0;
				m_words[wIndex] = new _Word(wid, tid, xid);
				
				m_xTopicSstat[xid][tid] ++;
				m_xSstat[xid] ++;
				if(xid ==1){
					m_localWordSstat[wid] ++;
					m_localWord ++;
				}
				
				
				wIndex ++;
			}
		}
	}
	

	public void createLocalWordTopicDistribution(int vocalbularySize, double beta){
		beta *= 0.01;
		m_localWordSstat = new double[vocalbularySize];
		m_localWordProb = new double[vocalbularySize];
		
		m_localWord = beta*vocalbularySize;
		Arrays.fill(m_localWordSstat, beta);
		Arrays.fill(m_localWordProb, 0);
		
	}
	
	public void estGlobalLocalTheta() {
		Utils.L1Normalization(m_xProportion);
		for(int x=0; x<m_xTopics.length; x++)
			Utils.L1Normalization(m_xTopics[x]);
		
		for(_Word w: m_words){
			Utils.L1Normalization(w.m_xProb);
		}
		
		Utils.L1Normalization(m_localWordProb);
	}
	
	public void collectLocalWordSstat() {
		for(int i=0; i<m_localWordSstat.length; i++){
			m_localWordProb[i] += m_localWordSstat[i];
		}
	}
	
	
}
