package structures;

public class _ChildDoc4DCMDMMCorrLDA extends _ChildDoc{
	
	public int m_topic;
	
	public _ChildDoc4DCMDMMCorrLDA(int ID, String name, String title, String source, int ylabel){
		super(ID, name, title, source, ylabel);
		
	}
	
	public void setTopics4Gibbs(int k, double alpha){
		createSpace(k, alpha);
		
		int tid = 0;
		tid = m_rand.nextInt(k);
		m_topic = tid;
		m_sstat[tid] ++;
	}
	
	
}
