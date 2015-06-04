package Classifier.semisupervised;

import structures._Doc;
import utils.Utils;

public class PairwiseSimCalculator implements Runnable {

	//pointer to the Gaussian Field object to calculate similarity in parallel
	GaussianFields m_GFObj; 
	int m_start, m_end;
//	boolean m_topicFlag;
	
	//in the range of [start, end)
	public PairwiseSimCalculator(GaussianFields obj, int start, int end, boolean topicFlag) {
		m_GFObj = obj;
		m_start = start;
		m_end = end;
//		m_topicFlag = topicFlag;
	}

	@Override
	public void run() {
		_Doc di, dj;
		double similarity=0;
		for (int i = m_start; i < m_end; i++) {
			di = m_GFObj.getTestDoc(i);
			for (int j = i + 1; j < m_GFObj.m_U; j++) {// to save computation since our similarity metric is symmetric
				dj = m_GFObj.getTestDoc(j);
				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				m_GFObj.setCache(i, j, similarity);
			}

			for (int j = 0; j < m_GFObj.m_L; j++) {
				dj = m_GFObj.m_labeled.get(j);
				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				m_GFObj.setCache(i, m_GFObj.m_U + j, similarity);
			}
			//set up the Y vector for unlabeled data
			m_GFObj.m_Y[i] = m_GFObj.m_classifier.predict(m_GFObj.getTestDoc(i)); //Multiple learner.
		}	
		System.out.format("[%d,%d) finished...\n", m_start, m_end);
	}
}