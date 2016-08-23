package structures;

import java.util.ArrayList;
import java.util.HashMap;

import utils.Utils;

public class _ChildDoc4DCMDMMCorrLDA extends _ChildDoc{
	
	public int m_topic;
	
	public _ChildDoc4DCMDMMCorrLDA(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		m_topic = -1;
	}
	
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		int tid = 0;
		tid = m_rand.nextInt(k);
		m_topic = tid;
		m_sstat[tid] ++;
		
		int wid = 0, wIndex = 0;
		for (_SparseFeature fv : m_x_sparse) {
			wid = fv.getIndex();
			for (int j = 0; j < fv.getValue(); j++) {
				m_words[wIndex] = new _Word(wid, tid);
				wIndex++;
			}
		}
	}
	
	public void createSparseVct() {
		HashMap<Integer, Double> inferVct = new HashMap<Integer, Double>();
		
		for (_Word w : m_words) {
			int wIndex = w.getIndex();
			int featureIndex = Utils.indexOf(m_x_sparse, wIndex);
			double featureVal = m_x_sparse[featureIndex].getValue();
			inferVct.put(wIndex, featureVal);
		}
		
		m_x_sparse_infer = Utils.createSpVct(inferVct);	
	}
	
	public void createSparseVct4Test() {
		createSparseVct4Infer();
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength) {
		int trainLength = m_totalLength - testLength;
		createSpace4GibbsTest(k, alpha, trainLength);
		m_testLength = testLength;
		m_testWords = new _Word[testLength];

		ArrayList<Integer> wordIndexs = new ArrayList<Integer>();
		while (wordIndexs.size() < testLength) {
			int testIndex = m_rand.nextInt(m_totalLength);
			if (!wordIndexs.contains(testIndex)) {
				wordIndexs.add(testIndex);
			}
		}

		int wIndex = 0, wTrainIndex = 0, wTestIndex = 0, tid, wid;
		tid = m_rand.nextInt(k);
		m_sstat[tid]++; // collect the topic proportion

		for (_SparseFeature fv : m_x_sparse) {
			wid = fv.getIndex();
			for (int j = 0; j < fv.getValue(); j++) {
				if (wordIndexs.contains(wIndex)) {
					m_testWords[wTestIndex] = new _Word(wid, tid);
					wTestIndex++;
				} else {
					m_words[wTrainIndex] = new _Word(wid, tid);
					wTrainIndex++;

				}

				wIndex++;
			}

		}

	}
}
