package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import utils.Utils;

public class _ParentDoc extends _Doc {
	////inherit from _Doc
	//public int[] m_words;
	//public int[] m_topicAssignment;
	public ArrayList<_ChildDoc> m_childDocs;
	
	public _ParentDoc(int ID, String name, String title, String source, int ylabel) {
		super(ID, source, ylabel);
		m_childDocs = new ArrayList<_ChildDoc>();
		setName(name);
		setTitle(title);
		// TODO Auto-generated constructor stub
	}
	
	
	public void addChildDoc(_ChildDoc cDoc){
		m_childDocs.add(cDoc);
	}
	
	//not set m_sstat to alpha
	public void setTopics4Gibbs(int k){
		if(m_topics == null || m_topics.length != k){
			m_topics = new double[k];
			m_sstat = new double[k];
		}
		
		Arrays.fill(m_sstat, 0);
		
		int docSize = (int)Utils.sumOfFeaturesL1(m_x_sparse);
		if(m_words==null||m_words.length!=docSize){
			m_topicAssignment = new int[docSize];
			m_words = new int[docSize];
		}
		
		int wIndex = 0;
		if(m_rand==null){
			m_rand = new Random();
		}
		
		for(_SparseFeature fv:m_x_sparse){
			for(int j=0; j<fv.getValue(); j++){
				m_words[wIndex] = fv.getIndex();
				m_topicAssignment[wIndex] = m_rand.nextInt(k);

				m_sstat[m_topicAssignment[wIndex]] ++;
				
				wIndex ++;
			}
		}
	}
	
//	public void  setTopics4Gibbs(int k, double alpha){
//	
//	}

}
