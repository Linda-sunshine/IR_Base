//////////////************used to store the parentDoc2***************////////////
package structures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class _ChildDoc2 extends _ChildDoc {

	public _ParentDoc2 m_parentDoc2;

	// //record the index of the token and remain constant
	public int[] m_index;
	public int[] m_positionInDoc;

	public _ChildDoc2(int ID, String name, String title, String source,
			int ylabel) {
		super(ID, name, title, source, ylabel);
		// m_parentDoc2 = null;
		m_parentDoc2 = null;
		m_index = null;

		// TODO Auto-generated constructor stub
	}

	public void setIndex(ArrayList<Integer> indexList) {
		int indexListLen = indexList.size();
		m_index = new int[indexListLen];
		m_positionInDoc = new int[indexListLen];
		for (int i = 0; i < indexListLen; i++) {
			m_index[i] = indexList.get(i);
			m_positionInDoc[i] = i;
		}
	}

	public void setParentDoc2(_ParentDoc2 pDoc) {
		m_parentDoc2 = pDoc;

	}

	public void setTopics4Gibbs(int k,
			double[] gamma) {
		int indicatorNum = gamma.length;
		if (m_topics == null || m_topics.length != k) {

			m_topics = new double[k];
			m_xTopicSstat = new double[indicatorNum][k];
			m_xTopics = new double[indicatorNum][k];
		}

		m_xSstat = new double[indicatorNum];
		m_xProportion = new double[indicatorNum];
		for (int i = 0; i < indicatorNum; i++) {
			m_xSstat[i] = 0;
			m_xProportion[i] = 0;
			Arrays.fill(m_xTopicSstat[i], 0);
		}

		// _SparseFeature[] x_sparse = getSparse();

		int docSize = getTotalDocLength();
		if (m_words == null || m_words.length != docSize) {
			m_words = new int[docSize];
			m_topicAssignment = new int[docSize];
			m_xIndicator = new int[docSize];
		}

		int wIndex = 0;
		if (m_rand == null) {
			m_rand = new Random();
		}
		
		for (int j = 0; j < getTotalDocLength(); j++) {
			m_words[j] = m_index[j];
			m_topicAssignment[j] = m_rand.nextInt(k);
			m_xIndicator[j] = m_rand.nextInt(indicatorNum);

			m_xTopicSstat[m_xIndicator[j]][m_topicAssignment[j]]++;
			m_xSstat[m_xIndicator[j]]++;

		}
		// System.out.println();

	}

	public void permutation(int flag) {
		super.permutation(flag);
	}
	
	// add swapping x_indicator
	public void permutation() {
		// System.out.println("permutation dddd");
		int s, t;
		for (int i = m_words.length - 1; i > 1; i--) {
			s = m_rand.nextInt(i);

			swapPosition(i, s);
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
	
	// position1 > position2
	public void swapPosition(int position1, int position2) {
		for (int i = 0; i < m_words.length; i++) {
			if (m_positionInDoc[i] == position2) {
				m_positionInDoc[i] = position1;
			}else{
				if (m_positionInDoc[i] == position1) {
					m_positionInDoc[i] = position2;
				}
			}
			

		}
	}

}
