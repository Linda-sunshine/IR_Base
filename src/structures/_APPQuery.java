package structures;

import java.util.ArrayList;
import java.util.HashMap;

public class _APPQuery {
	double m_score;
	double m_relevance;
	_Word[] m_Words;
	String m_relevantAPP;
	int m_queryID;
	HashMap<_ParentDoc, Double> m_returnedAPP;
	
	public _APPQuery(){
		m_score = 0;
		m_relevantAPP = "google play";
		m_queryID = -1;
		m_returnedAPP = new HashMap<_ParentDoc, Double>();
	}
	
	public _APPQuery(int queryID){
		m_score = 0;
		m_relevantAPP = "google play";
		m_queryID = queryID;
		m_returnedAPP = new HashMap<_ParentDoc, Double>();
	}
	
	public void initWords(ArrayList<_Word> wordList){
		m_Words = new _Word[wordList.size()];
		int i=0;
		for(_Word w: wordList){
			m_Words[i] = new _Word(w.getIndex());
		}
		
	}
	
	public int getQueryID(){
		return m_queryID;
	}
	
	public void setScore(double score){
		m_score = score;
	}
	
	public double getScore(){
		return m_score;
	}
	
	public void setRelevance(double relevance){
		m_relevance = relevance;
	}
	
	public double getRelevance(){
		return m_relevance;
	}
	
	public void setRelevantAPP(String APPName){
		m_relevantAPP = APPName;
	}
	
	public String getRelevantAPP(){
		return m_relevantAPP;
	}
	
	public _Word[] getWords(){
		return m_Words;
	}
	
	public void addReturnedAPP(){
//		m_returnedAPP.put
	}
}
