package structures;

public class _Doc4SVMTopic extends _Doc{
	/////0--not a doc for test4SVM
	////1---a doc for test4SVM
	protected int m_test4SVMFlag;
	
	public _Doc4SVMTopic(int ID, String source, int ylabel){
		super(ID, source, ylabel);
		m_test4SVMFlag = 0;
	}
	
	public int getTest4SVMFlag(){
		return m_test4SVMFlag;
	}
	
	public void setTest4SVMFlag(int flag){
		m_test4SVMFlag = flag;
	}
}
