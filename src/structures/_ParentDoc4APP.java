package structures;

import java.util.ArrayList;

public class _ParentDoc4APP extends _ParentDoc{

	public _ParentDoc4APP(int ID, String name, String title, String source, int ylabel) {
		super(ID, name, title, source, ylabel);

		m_childDocs = new ArrayList<_ChildDoc>();

		setName(name);
		setTitle(title);
	}
	
	@Override
	public void setTopics4Gibbs(int k, double alpha) {
		createSpace(k, alpha);
		
		int wIndex = 0, wid, tid;
		for(_SparseFeature fv:m_x_sparse) {
			wid = fv.getIndex();
			for(int j=0; j<fv.getValue(); j++) {
				tid = m_rand.nextInt(k);
				m_words[wIndex] = new _Word(wid, tid);// randomly initializing the topics inside a document
				m_sstat[tid] ++; // collect the topic proportion
						
				wIndex ++;
			}
		}
		
	}
	
	public void setTopics4GibbsTest(int k, double alpha, int testLength){
		super.setTopics4GibbsTest(k, alpha, testLength);
	}
	
}
