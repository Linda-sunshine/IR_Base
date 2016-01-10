package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import utils.Utils;

public class _ParentDoc2 extends _ParentDoc {
	// //inherit from _Doc
	// public int[] m_words;
	// public int[] m_topicAssignment;
	public ArrayList<_ChildDoc2> m_childDocs;

	public HashMap<Integer, _Stn> m_sentenceMap;

	public int[] m_sentenceSize;

	public _ParentDoc2(int ID, String name, String title, String source,
			int ylabel) {
		super(ID, name, title, source, ylabel);
		m_childDocs = new ArrayList<_ChildDoc2>();
		m_sentenceMap = new HashMap<Integer, _Stn>();
		setName(name);
		setTitle(title);

		// TODO Auto-generated constructor stub
	}

	public void addChildDoc(_ChildDoc2 cDoc) {
		m_childDocs.add(cDoc);
	}

	// not set m_sstat to alpha
	public void setTopics4Gibbs(int k) {
		if (m_topics == null || m_topics.length != k) {
			m_topics = new double[k];
			m_sstat = new double[k];
		}

		Arrays.fill(m_sstat, 0);

		int docSize = (int) Utils.sumOfFeaturesL1(m_x_sparse);
		if (m_words == null || m_words.length != docSize) {
			m_topicAssignment = new int[docSize];
			m_words = new int[docSize];
		}

		int wIndex = 0;
		if (m_rand == null) {
			m_rand = new Random();
		}

		// System.out.println("docSize\t" + docSize);
		for (int i=0; i<m_sentenceMap.size(); i++) {
			_Stn stnObject = m_sentenceMap.get(i);


			for (int j = 0; j < stnObject.m_stnLength; j++) {
				// System.out.println("wIndex\t" + wIndex);
				// int p = m_words[wIndex];
				// int q = stnObject.m_words[j];
				//
				m_words[wIndex] = stnObject.m_words[j];
				 m_topicAssignment[wIndex] = m_rand.nextInt(k);
				
				 m_sstat[m_topicAssignment[wIndex]]++;

				wIndex++;
			}
		}
		// m_words[wIndex - 1] = 1;
	}

	public void permutation() {
		int s, t;

		for (int i = m_words.length - 1; i > 1; i--) {
			s = m_rand.nextInt(i);

			swapWordIndexInSentence(s, i);

			// swap the word
			t = m_words[s];
			m_words[s] = m_words[i];
			m_words[i] = t;

			// swap the topic assignment
			t = m_topicAssignment[s];
			m_topicAssignment[s] = m_topicAssignment[i];
			m_topicAssignment[i] = t;
		}

	}

	// word1 < word2;
	public void swapWordIndexInSentence(int wordIndexInDoc1, int wordIndexInDoc2) {
//		for (int sentenceID : m_sentenceMap.keySet()) {
		for (int s = 0; s < m_sentenceMap.size(); s++) {
			int[] indexInDoc = m_sentenceMap.get(s).m_wordPositionInDoc;
			// String[] tokens = m_sentenceMap.get(s).m_;
			for(int i=0; i<indexInDoc.length; i++){
				if (indexInDoc[i] == wordIndexInDoc1) {
					indexInDoc[i] = wordIndexInDoc2;
				}else{
					if (indexInDoc[i] == wordIndexInDoc2) {
						indexInDoc[i] = wordIndexInDoc1;
					}
				}
			}
		}
		
	}
	
}
