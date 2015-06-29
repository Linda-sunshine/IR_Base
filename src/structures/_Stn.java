/**
 * 
 */
package structures;

import java.util.Set;

/**
 * @author hongning
 * Sentence structure for text documents
 */
public class _Stn {
	_SparseFeature[] m_x_sparse; // bag of words for a sentence
	
	//structure for HTMMM
	double[] m_transitFv; // features for determine topic transition
	double m_transitStat; // posterior topic transit probability

	// tructure for LR-HTSM
	double[] m_sentiTransitFv; // features for determine sentiment transition
	double m_sentiTransitStat; // posterior sentiment transit probability
	
	String[] m_sentencePOSTag;
	
	//structure for topic assignment
	int m_topic; //topic/aspect assignment
	
	//attribute label for NewEgg data
	int m_label = 0; // default is neutral
	
	public _Stn(_SparseFeature[] x) {
		m_x_sparse = x;
		
		m_transitFv = new double[_Doc.stn_fv_size];
		m_sentiTransitFv = new double[_Doc.stn_senti_fv_size];
	}
	
	public _Stn(_SparseFeature[] x, int label) {
		m_x_sparse = x;
		m_label = label;
		m_transitFv = new double[_Doc.stn_fv_size];
		m_sentiTransitFv = new double[_Doc.stn_senti_fv_size];
	}

	public _SparseFeature[] getFv() {
		return m_x_sparse;
	}
	
	public void setSentenceLabel(int label){
		m_label = label;
	}

	public void setSentencePosTag(String[] sentenceposTags){
		m_sentencePOSTag = sentenceposTags;
	}
	
	public String[] getSentencePosTag(){
		return m_sentencePOSTag;
	}	
	
	public int getSentenceLabel(){
		return m_label;
	}
	
	public double[] getTransitFvs() {
		return m_transitFv;
	}
	
	public double[] getSentiTransitFvs() {
		return m_sentiTransitFv;
	}
	
	public double getTransitStat() {
		return m_transitStat;
	}
	
	public void setTransitStat(double t) {
		m_transitStat = t;
	}
	
	public double getSentiTransitStat() {
		return m_sentiTransitStat;
	}
	
	public void setSentiTransitStat(double t) {
		m_sentiTransitStat = t;
	}
	
	public int getTopic() {
		return m_topic;
	}
	
	public void setTopic(int i) {
		m_topic = i;
	}
	
	//annotate by all the words
	public int AnnotateByKeyword(Set<Integer> keywords){
		int count = 0;
		for(_SparseFeature t:m_x_sparse){
			if (keywords.contains(t.getIndex()))
				count ++;
		}
		return count;
	}
}
