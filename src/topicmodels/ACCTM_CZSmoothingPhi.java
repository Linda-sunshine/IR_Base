package topicmodels;

import java.util.HashMap;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._ParentDoc;
import structures._Word;

public class ACCTM_CZSmoothingPhi extends ACCTM_CZ{

	public ACCTM_CZSmoothingPhi(int number_of_iteration, double converge, double beta, _Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, ksi, tau);
		
	}
	
	public String toString(){
		return String.format("ACCTM_CZ topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected double childLocalWordByTopicProb(int wid, _ChildDoc4BaseWithPhi d){
		double smoothingProb = LMSmoothingProb(wid);
		return (d.m_xTopicSstat[1][wid]+d_beta*smoothingProb)/(d.m_childWordSstat+d_beta);
	}
	
	protected void collectChildStats(_ChildDoc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		_ParentDoc pDoc = cDoc.m_parentDoc;
		double parentDocLength = pDoc.getDocInferLength();
				
		for (int k = 0; k < this.number_of_topics; k++) 
			cDoc.m_xTopics[0][k] += cDoc.m_xTopicSstat[0][k] + d_alpha+cDoc.getMu()*pDoc.m_sstat[k] / parentDocLength;
		
		for(int x=0; x<m_gamma.length; x++)
			cDoc.m_xProportion[x] += m_gamma[x] + cDoc.m_xSstat[x];
		
		for(int w=0; w<vocabulary_size; w++){
			double smoothingProb = LMSmoothingProb(w);
			cDoc.m_xTopics[1][w] += cDoc.m_xTopicSstat[1][w]+d_beta*smoothingProb;
		}
	
		for(_Word w:d.getWords()){
			w.collectXStats();
		}
	}
	
	protected double LMSmoothingProb(int wid){
		if(!m_wordSstat.containsKey(wid)){
			return 0;
		}else
			return m_wordSstat.get(wid);
	}
	
	
}
