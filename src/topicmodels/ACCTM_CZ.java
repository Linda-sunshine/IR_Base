package topicmodels;

import java.util.ArrayList;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._SparseFeature;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_CZ extends ParentChildBaseWithPhi_Gibbs{
	public ACCTM_CZ(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, ksi, tau);
	}
	
	public String toString(){
		return String.format("ACCTM_CZ topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected double parentChildInfluenceProb(int tid, _ParentDoc pDoc){
		double term = 1.0;
		
		if(m_collectCorpusStats){
			return term;
		}
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			term *= influenceRatio(pDoc.m_sstat[tid], pDoc.m_sstat[0]);
		}
		
		return term;
	}
	
	protected double influenceRatio(double njc, double n1c){
		double ratio = 1.0;
		
		for(int n=1; n<=n1c; n++){
			ratio *= n1c*1.0/(n1c+1);
		}
		
		for(int n=1; n<=njc; n++){
			ratio *= (njc+1)*1.0/njc;
		}
		
		return ratio;
	}
	
	protected double childTopicInDocProb(int tid, _ChildDoc d){
		double docLength = d.m_parentDoc.getDocInferLength();
		
		return (d.m_parentDoc.m_sstat[tid]/docLength);
	}
	
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
	
	protected void estThetaInDoc(_Doc d){
		if(d instanceof _ParentDoc){
			Utils.L1Normalization(d.m_topics);
		}else if(d instanceof _ChildDoc4BaseWithPhi){
			((_ChildDoc4BaseWithPhi)d).estGlobalLocalTheta();
		}
		
		m_statisticsNormalized = true;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc) d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = (int)pDoc.getTotalDocLength();
		pDoc.setTopics4GibbsTest(number_of_topics, 0, 0);
	
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
		
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			
		}
	}
	
//	protected double testLogLikelihoodByIntegrateTopics(_ChildDoc d) {
//		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi) d;
//		double docLogLikelihood = 0.0;
//		double gammaLen = Utils.sumOfArray(m_gamma);
//
//		// prepare compute the normalizers
//		_SparseFeature[] fv = cDoc.getSparse();
//
//		for (_Word w : cDoc.getTestWords()) {
//			int wid = w.getIndex();
//
//			double wordLogLikelihood = 0;
//			for (int k = 0; k < number_of_topics; k++) {
//				double wordPerTopicLikelihood = childWordByTopicProb(k, wid)
//						* childTopicInDocProb(k, cDoc)
//						* childXInDocProb(0, cDoc)
//						/ (cDoc.getDocInferLength() + gammaLen);
//				wordLogLikelihood += wordPerTopicLikelihood;
//			}
//			double wordPerTopicLikelihood = childLocalWordByTopicProb(wid, cDoc)
//					* childXInDocProb(1, cDoc)
//					/ (cDoc.getDocInferLength() + gammaLen);
//			wordLogLikelihood += wordPerTopicLikelihood;
//
//			if (Math.abs(wordLogLikelihood) < 1e-10) {
//				System.out.println("wordLoglikelihood\t" + wordLogLikelihood);
//				wordLogLikelihood += 1e-10;
//			}
//
//			wordLogLikelihood = Math.log(wordLogLikelihood);
//			docLogLikelihood += wordLogLikelihood;
//		}
//
//		return docLogLikelihood;
//	}
	
}
