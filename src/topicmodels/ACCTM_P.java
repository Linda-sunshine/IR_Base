package topicmodels;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

import structures._ChildDoc;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._Stn;
import structures._Word;
import utils.Utils;

public class ACCTM_P extends ACCTM_C {
	public ACCTM_P(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag, gamma, ksi, tau);

	}
	
	public String toString() {
		return String.format("ACCTM_P topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}

	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);
		for(_Doc d:collection){
			if(d instanceof _ParentDoc)
				((_ParentDoc)d).initWordDistribution(vocabulary_size, d_beta);
		}
	}
	
	protected double childLocalWordByTopicProb(int wid, _ChildDoc4BaseWithPhi d){
		_ParentDoc pDoc = d.m_parentDoc;
		return pDoc.getWordDistribution(wid);
	}
	
	public void initTest4Spam(ArrayList<_Doc> sampleTestSet, _Doc d) {
		_ParentDoc pDoc = (_ParentDoc)d;
		pDoc.setTopics4Gibbs(number_of_topics, 0);
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}

		sampleTestSet.add(pDoc);
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4Gibbs(number_of_topics, 0);
			sampleTestSet.add(cDoc);
			cDoc.setParentDoc(pDoc);
			computeMu4Doc(cDoc);
		}
		
		pDoc.initWordDistribution(vocabulary_size, d_beta);
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		double gammaLen = Utils.sumOfArray(m_gamma);

		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		for(_ChildDoc d:pDoc.m_childDocs){
			_ChildDoc4BaseWithPhi cDoc =(_ChildDoc4BaseWithPhi)d;
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc)*childXInDocProb(0, cDoc)/(gammaLen+cDoc.getDocInferLength());
//					wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDocProb(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				wordLogLikelihood += childLocalWordByTopicProb(wid, cDoc)*childXInDocProb(1, cDoc)/(gammaLen+cDoc.getDocInferLength());
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
}
