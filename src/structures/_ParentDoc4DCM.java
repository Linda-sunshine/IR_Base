package structures;

import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class _ParentDoc4DCM extends _ParentDoc {
	
	public double[][] m_wordTopic_stat;
	public double[][] m_wordTopic_prob;
	public double[] m_topic_stat;
	public double[][] m_lambda_stat;
	public double[] m_lambda_topicStat;
	
	public _ParentDoc4DCM(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		
	}

	public void setTopics4Variational(int k, double alpha, int vocalSize, double beta){
		if (m_topics==null || m_topics.length!=k) {
			m_topics = new double[k];
			m_sstat = new double[k];//used as p(z|w,\phi)
			m_phi = new double[m_x_sparse.length][k]; // this is the eta in the derivation for variational inference
		}

		for(int i=0; i<m_x_sparse.length; i++)
			Arrays.fill(m_phi[i], 0);

		m_lambda_stat = new double[k][vocalSize];
		m_lambda_topicStat = new double[k];

		m_wordTopic_prob = new double[k][vocalSize];

		for(int i=0; i<k; i++) {
			Arrays.fill(m_lambda_stat[i], beta);
			Arrays.fill(m_wordTopic_prob[i], 0);
			m_lambda_topicStat[i] = Utils.sumOfArray(m_lambda_stat[i]);
		}

		Arrays.fill(m_sstat, alpha);
		Arrays.fill(m_topics, 0);
		for(int n=0; n<m_x_sparse.length; n++) {
			Utils.randomize(m_phi[n], alpha);
			double v = m_x_sparse[n].getValue();
			int wId = m_x_sparse[n].getIndex();
			for(int i=0; i<k; i++) {
				m_sstat[i] += m_phi[n][i] * v;
				m_lambda_stat[i][wId] += v*m_phi[n][i];
			}
		}

		for(int i=0; i<k; i++) {
			m_lambda_topicStat[i] = Utils.sumOfArray(m_lambda_stat[i]);
		}
	}

	protected void setWordTopicStat(int k, int vocalSize){
		m_topic_stat = new double[k];
		Arrays.fill(m_topic_stat, 0);
		
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
				m_topic_stat[tid] ++;
				wIndex ++;
			}
		}
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength, int vocalSize){
		int trainLength = m_totalLength-testLength;
		
		createSpace4GibbsTest(k, alpha, trainLength);
		setWordTopicStat(k, vocalSize);
		
		m_testLength = testLength;
		m_testWords = new _Word[testLength];
		
		ArrayList<Integer> wordIndexs = new ArrayList<Integer>();
		while(wordIndexs.size()<testLength){
			int testIndex = m_rand.nextInt(m_totalLength);
			if(!wordIndexs.contains(testIndex)){
				wordIndexs.add(testIndex);
			}
		}
		
		int wIndex = 0, wTrainIndex = 0, wTestIndex = 0, tid=0, wid=0;
		for(_SparseFeature sf:m_x_sparse){
			wid = sf.getIndex();
			for(int j=0; j<sf.getValue(); j++){
				if(wordIndexs.contains(wIndex)){
					tid = m_rand.nextInt(k);
					m_testWords[wTestIndex] = new _Word(wid, tid);
					wTestIndex ++;
				}else{
					tid = m_rand.nextInt(k);
					m_words[wTrainIndex] = new _Word(wid, tid);
					wTrainIndex ++;
					m_wordTopic_stat[tid][wid] ++;
					m_topic_stat[tid] ++;
					m_sstat[tid] ++;
				}
				wIndex ++;
			}
		}
		
		m_phi = new double[m_x_sparse.length][k];
		m_word2Index = new HashMap<Integer, Integer>();
		for(int i=0; i<m_x_sparse.length; i++) 
			m_word2Index.put(m_x_sparse[i].m_index, i);
	}
	
}
