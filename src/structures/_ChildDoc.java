package structures;

import java.util.Arrays;
import java.util.Random;

public class _ChildDoc extends _Doc {
	public int[] m_xIndicator;
	public double[][] m_xTopicSstat;//0 from general, 1 from specific
	public double[] m_xSstat;
	public _ParentDoc m_parentDoc;
	

	public _ChildDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);
		m_parentDoc = null;
		setName(name);
		setTitle(title);
		// TODO Auto-generated constructor stub
	}
	
	public void setParentDoc(_ParentDoc pDoc){
		m_parentDoc = pDoc;
	}
	
	public void setTopics4Gibbs(int k, int indicatorNum, double[] gamma){
		if(m_topics==null||m_topics.length != k){
			m_topics = new double[k];
			m_xTopicSstat = new double[indicatorNum][k];
		}
		
		m_xSstat = new double[indicatorNum];
		
		for(int i=0; i<indicatorNum; i++){
			m_xSstat[i] = 0;
			Arrays.fill(m_xTopicSstat[i], 0);
		}

		
		//_SparseFeature[] x_sparse = getSparse();
		
		int docSize = getTotalDocLength();
		if(m_words==null||m_words.length!=docSize){
			m_words = new int[docSize];
			m_topicAssignment = new int[docSize];	
			m_xIndicator = new int[docSize];
		}
		
		int wIndex = 0;
		if(m_rand == null){
			m_rand =  new Random();
		}
		
		for(_SparseFeature fv: m_x_sparse){
			for(int j=0; j<fv.getValue(); j++){
				m_words[wIndex] = fv.getIndex();
				m_topicAssignment[wIndex] = m_rand.nextInt(k);
				m_xIndicator[wIndex] = m_rand.nextInt(indicatorNum);

				m_xTopicSstat[m_xIndicator[wIndex]][m_topicAssignment[wIndex]] ++;
				m_xSstat[m_xIndicator[wIndex]] ++;
						
				wIndex += 1;
			}
		}
	}
	
	//add swapping x_indicator
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
			
			t = m_xIndicator[s];
			m_xIndicator[s] = m_xIndicator[i];
			m_xIndicator[i] = t;
		}
	}
	
}
