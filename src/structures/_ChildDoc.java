package structures;

import java.util.Arrays;
import java.util.Random;

public class _ChildDoc extends _Doc {
	public int[] m_xIndicator;
	public double[][] m_xTopicSstat;//joint assignment of <x,z>: 0 from general, 1 from specific
	public double[] m_xSstat; // sufficient statistics for x

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
		m_xTopicSstat = new double[gammaSize][k];
		m_xTopics = new double[gammaSize][k];
		m_xSstat = new double[gammaSize];
		m_xProportion = new double[gammaSize];
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha){		
		if(m_topics==null||m_topics.length != k){		
			m_topics = new double[k];
			m_sstat = new double[k];
		}
	
		Arrays.fill(m_sstat, alpha);
		
		int docSize = getTotalDocLength();
		if(m_words==null || m_words.length!=docSize){
			m_words = new int[docSize];
			m_topicAssignment = new int[docSize];	
			m_xIndicator = new int[docSize];
		}
		
		int wIndex = 0;
		if(m_rand == null)
			m_rand =  new Random();
		
		for(_SparseFeature fv: m_x_sparse){
			for(int j=0; j<fv.getValue(); j++){
				m_words[wIndex] = fv.getIndex();
				m_topicAssignment[wIndex] = m_rand.nextInt(k);
				m_xIndicator[wIndex] = m_rand.nextInt(m_xSstat.length);

				m_xTopicSstat[m_xIndicator[wIndex]][m_topicAssignment[wIndex]] ++;
				m_xSstat[m_xIndicator[wIndex]] ++;
						
				wIndex += 1;
			}
		}
	}

	//add swapping x_indicator
	@Override
	public void permutation(){
		int s, t;
		for(int i=m_words.length-1; i>1; i--){
			s = m_rand.nextInt(i);
			
			t = m_words[s];
			m_words[s] = m_words[i];
			m_words[i] = t;
			
			t = m_topicAssignment[s];
			m_topicAssignment[s] = m_topicAssignment[i];
			m_topicAssignment[i] = t;
			
 			if (m_xIndicator!=null) {
				t = m_xIndicator[s];
				m_xIndicator[s] = m_xIndicator[i];
				m_xIndicator[i] = t;
 			}
		}
	}	
}
