package topicmodels.correspondenceModels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_CZ extends ACCTM_C{
	public ACCTM_CZ(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma);
	}
	
	@Override
	public String toString(){
		return String.format("ACCTM_CZ topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	@Override
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			term *= influenceRatio(cDoc.m_xTopicSstat[0][tid], pDoc.m_sstat[tid], cDoc.m_xTopicSstat[0][0], pDoc.m_sstat[0]);
		}
		
		return term;
	}
	
	protected double influenceRatio(double njc, double njp, double n1c, double n1p){
		double ratio = 1.0;
		double smoothingParameter = 1e-20;
		
		for(int n=1; n<=n1c; n++){
			ratio *= (n1p + smoothingParameter) * 1.0
					/ (n1p + 1 + smoothingParameter);
		}
		
		for(int n=1; n<=njc; n++){
			ratio *= (njp + 1 + smoothingParameter) * 1.0
					/ (njp + smoothingParameter);
		}
		
		return ratio;
	}
	
	@Override
	protected double childTopicInDocProb(int tid, _ChildDoc d){
		double smoothingParameter = 1e-20;
		double docLength = d.m_parentDoc.getDocInferLength();
		
		return (d.m_parentDoc.m_sstat[tid]+smoothingParameter)/(docLength+smoothingParameter*number_of_topics);
	}
	
	@Override
	protected void collectChildStats(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		_ParentDoc pDoc = cDoc.m_parentDoc;
		double parentDocLength = pDoc.getDocInferLength();
		
		for(int k=0; k<number_of_topics; k++){
			cDoc.m_xTopics[0][k] += cDoc.m_xTopicSstat[0][k];
		}
		
		for(int x=0; x<m_gamma.length; x++){
			cDoc.m_xProportion[x] += m_gamma[x]+cDoc.m_xSstat[x];
		}
		
		for(int w=0; w<vocabulary_size; w++){
			cDoc.m_xTopics[1][w] += cDoc.m_xTopicSstat[1][w];
		}
		
		for(_Word w:d.getWords()){
			w.collectXStats();
		}
	}
		
	@Override
	public double inference(_Doc pDoc){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
		return inference4Doc(sampleTestSet);	
	}
	
	@Override
	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d) {
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
		double docLogLikelihood = 0.0;
		double gammaLen = Utils.sumOfArray(m_gamma);

		for (_Word w : cDoc.getTestWords()) {
			int wid = w.getIndex();

			double wordLogLikelihood = 0;
			for (int k = 0; k < number_of_topics; k++) {
				double term1 = childWordByTopicProb(k, wid);
				double term2 = childTopicInDoc(k, cDoc);
				double term3 = childXInDocProb(0, cDoc)/ (cDoc.getDocInferLength() + gammaLen);
				
				double wordPerTopicLikelihood = term1*term2*term3;
				wordLogLikelihood += wordPerTopicLikelihood;
			}
			double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)
					* childXInDocProb(1, cDoc)
					/ (cDoc.getDocInferLength() + gammaLen);
			wordLogLikelihood += wordPerTopicLikelihood;

			if (Math.abs(wordLogLikelihood) < 1e-10) {
				System.out.println("wordLoglikelihood\t" + wordLogLikelihood);
				wordLogLikelihood += 1e-10;
			}

			wordLogLikelihood = Math.log(wordLogLikelihood);
			docLogLikelihood += wordLogLikelihood;
		}

		return docLogLikelihood;
	}

	public double childTopicInDoc(int tid, _ChildDoc cDoc){
//		System.out.println("number of words in tid\t"+cDoc.m_sstat[tid]+"\t x=0 words \t"+cDoc.m_xSstat[0]);
		return (cDoc.m_xTopicSstat[0][tid]+1e-10)/(cDoc.m_xSstat[0]+1e-10*number_of_topics);
	}
}
