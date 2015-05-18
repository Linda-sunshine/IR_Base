package Classifier.semisupervised;

import structures._Doc;
import utils.Utils;

public class PairwiseSimCalculator implements Runnable {

	//pointer to the Gaussian Field object to calculate similarity in parallel
	GaussianFields m_GFObj; 
	int m_start, m_end;
	boolean m_topicFlag;
	
	//in the range of [start, end)
	public PairwiseSimCalculator(GaussianFields obj, int start, int end, boolean topicFlag) {
		m_GFObj = obj;
		m_start = start;
		m_end = end;
		m_topicFlag = topicFlag;
	}

	@Override
	public void run() {
		_Doc di, dj;
		double similarity, discount;
		for (int i = m_start; i < m_end; i++) {
			di = m_GFObj.getTestDoc(i);
			for (int j = i + 1; j < m_GFObj.m_U; j++) {// to save computation since our similarity metric is symmetric
//				similarity = 0; 
				dj = m_GFObj.getTestDoc(j);
				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if(m_topicFlag){//If it is topic based similarity.
					double tmp = Math.exp(m_GFObj.getTopicSimilarity(di, dj));
					if(!Double.isInfinite(tmp))
						similarity += tmp;	
					else 
						System.out.println("Infinity detected");
				}
//				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
//				if(m_topicFlag){
//					if (di.sameProduct(dj)){
//						double aspScore = Utils.dotProduct(di.getAspVct(), dj.getAspVct());
//						if(aspScore != 0)
//							discount = Math.pow(1.5, aspScore);
//						else 
//							discount = m_GFObj.m_discount;
//						similarity *= discount;
//					}
//				}	
				m_GFObj.setCache(i, j, similarity);
			}

			for (int j = 0; j < m_GFObj.m_L; j++) {
				dj = m_GFObj.m_labeled.get(j);
//				similarity = 0;
				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
				if(m_topicFlag){//If it is topic based similarity.
					double tmp = Math.exp(m_GFObj.getTopicSimilarity(di, dj));
					if(!Double.isInfinite(tmp))
						similarity += tmp;
					else 
						System.out.println("Infinity detected");
				}
//				similarity = m_GFObj.getSimilarity(di, dj) * di.getWeight() * dj.getWeight();
//				if(m_topicFlag){
//					if (di.sameProduct(dj)){
//						double aspScore = Utils.dotProduct(di.getAspVct(), dj.getAspVct());
//						if(aspScore != 0)
//							discount = Math.pow(1.5, aspScore);
//						else 
//							discount = m_GFObj.m_discount;
//						similarity *= discount;
//					}
//				}	
				m_GFObj.setCache(i, m_GFObj.m_U + j, similarity);
			}
			
			//set up the Y vector for unlabeled data
			m_GFObj.m_Y[i] = m_GFObj.m_classifier.predict(m_GFObj.getTestDoc(i)); //Multiple learner.
		}	
		System.out.format("[%d,%d) finished...\n", m_start, m_end);
	}

}
