package structures;

import java.util.ArrayList;
import java.util.Collections;

import Classifier.supervised.liblinear.Feature;

public class _Query {
	public ArrayList<_QUPair> m_docList;
	int m_pairSize;
	
	public _Query(){
		m_docList = new ArrayList<_QUPair>();
		m_pairSize = 0;
	}
	
	public void addQUPair(_QUPair pair){ m_docList.add(pair); }
	
	public int createRankingPairs() {
		_QUPair qui, quj;
		for(int i=0; i<m_docList.size(); i++) {
			qui = m_docList.get(i);
			for(int j=0; j<m_docList.size(); j++) {
				if (i == j)
					continue;
				quj = m_docList.get(j);
				
				if (qui.m_y > quj.m_y) {
					qui.addWorseURL(quj);
					quj.addBetterURL(qui);
					m_pairSize ++;
				} else if (qui.m_y < quj.m_y) {
					qui.addBetterURL(quj);
					quj.addWorseURL(qui);
					m_pairSize ++;
				}
			}
		}
		return m_pairSize;
	}
	
	public void sortDocs() {
		Collections.sort(m_docList);//sort documents by predicted ranking score
	}
	
	public int getPairSize() {
		return m_pairSize;
	}
	
	public int getDocSize() {
		return m_docList.size();
	}
	
	public void extractPairs4RankSVM(ArrayList<Feature[]> fvs, ArrayList<Integer> labels) {
		for(_QUPair di:m_docList) {
			if (di.m_worseURLs==null)
				continue;
			
			for(_QUPair dj:di.m_worseURLs) {
				fvs.add(di.getDiffFv(dj));
				labels.add(1);
			}
		}
	}
}
