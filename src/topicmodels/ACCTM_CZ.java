package topicmodels;

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
		
		if(tid==0)
			return term;
		
		for(_ChildDoc cDoc:pDoc.m_childDocs4Dynamic){
			term *= influenceRatio(cDoc.m_xTopicSstat[0][tid], pDoc.m_sstat[tid], cDoc.m_xTopicSstat[0][0], pDoc.m_sstat[0]);
		}
		
		return term;
	}
	
	protected double influenceRatio(double njc, double njp, double n1c, double n1p){
		double ratio = 1.0;
		
		for(int n=1; n<=n1c; n++){
			ratio *= n1p*1.0/(n1p+1);
		}
		
		for(int n=1; n<=njc; n++){
			ratio *= (njp+1)*1.0/njp;
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
	

	public void printChildLikelihood4ParentSpam(){
		String childLikelihood4ParentSpamFile = "./data/results/dynamic/childLikelihood4ParentSample.txt";
		try{
			PrintWriter pw = new PrintWriter(new File(childLikelihood4ParentSpamFile));
			for(_Doc doc:m_testSet){
				_ParentDoc pDoc = (_ParentDoc)doc;
				pw.print(pDoc.getName()+"\t");
				for(_ChildDoc cDoc:pDoc.m_childDocs){
					double likelihood = computeChildLikelihood4ParentSpam(cDoc, pDoc);
					pw.print(cDoc.getName()+":"+likelihood+"\t");
				}
				pw.println();
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public double computeChildLikelihood4ParentSpam(_ChildDoc cDoc, _ParentDoc pDoc){
		double likelihood = 0;
		
		for(_Word w:cDoc.getWords()){
			
		}
		
		return likelihood;
	}
		
	//dynamical add comments to sampleTest
//	public void initTest4Dynamical(ArrayList<_Doc> sampleTestSet, _Doc d, int commentNum){
//		_ParentDoc pDoc = (_ParentDoc)d;
//		pDoc.m_childDocs4Dynamic = new ArrayList<_ChildDoc>();
//		pDoc.setTopics4Gibbs(number_of_topics, 0);
//		for(_Stn stnObj: pDoc.getSentences()){
//			stnObj.setTopicsVct(number_of_topics);
//		}
////		int testLength = (int)pDoc.getTotalDocLength();
////		testLength = 0;
////		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
//		
//		sampleTestSet.add(pDoc);
//		int count = 0;
//		for(_ChildDoc cDoc:pDoc.m_childDocs){
//			if(count>=commentNum){
//				break;
//			}
//			count ++;
//			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
//			
//			((_ChildDoc4BaseWithPhi)cDoc).setTopics4Gibbs(number_of_topics, 0);
//			sampleTestSet.add(cDoc);
//			pDoc.addChildDoc4Dynamics(cDoc);
//		}
//	}
	
	public double inference(_Doc pDoc){
		ArrayList<_Doc> sampleTestSet = new ArrayList<_Doc>();
		
		initTest(sampleTestSet, pDoc);
		return inference4Doc(sampleTestSet);	
	}
	
	public double inference4Doc(ArrayList<_Doc> sampleTestSet){
		double logLikelihood = 0.0, count = 0;
		int  iter = 0;
		do {
			int t;
			_Doc tmpDoc;
			for(int i=sampleTestSet.size()-1; i>1; i--) {
				t = m_rand.nextInt(i);
				
				tmpDoc = sampleTestSet.get(i);
				sampleTestSet.set(i, sampleTestSet.get(t));
				sampleTestSet.set(t, tmpDoc);			
			}
			
			for(_Doc doc: sampleTestSet)
				calculate_E_step(doc);
			
			if (iter>m_burnIn && iter%m_lag==0){
				for(_Doc doc: sampleTestSet){
					collectStats(doc);
				}
				count ++;
			}
		} while (++iter<this.number_of_iteration);
	
		for(_Doc doc: sampleTestSet){
			estThetaInDoc(doc);
//			logLikelihood += calculate_test_log_likelihood(doc);
		}
		
		return logLikelihood;
	}
	
	protected void initTest(ArrayList<_Doc> sampleTestSet, _Doc d){
		_ParentDoc pDoc = (_ParentDoc) d;
		for(_Stn stnObj: pDoc.getSentences()){
			stnObj.setTopicsVct(number_of_topics);
		}
		
		int testLength = (int)pDoc.getTotalDocLength();
//		testLength = 0;
		pDoc.setTopics4GibbsTest(number_of_topics, 0, testLength);
	
		sampleTestSet.add(pDoc);
		
		for(_ChildDoc cDoc:pDoc.m_childDocs4Dynamic){
			testLength = (int)(m_testWord4PerplexityProportion*cDoc.getTotalDocLength());
			((_ChildDoc4BaseWithPhi)cDoc).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
		
			((_ChildDoc4BaseWithPhi)cDoc).setTopics4GibbsTest(number_of_topics, 0, testLength);
			sampleTestSet.add(cDoc);
			
		}
	}
	
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
	
	protected HashMap<String, Double> rankChild4StnByHybrid(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getChildDocLenWithXVal();
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			for(_Word w:stnObj.getWords()){
				double featureLikelihood = 0;
				
				int wid = w.getIndex();
				double docVal = 0;
				if(cDoc.m_wordXStat.containsKey(wid)){
					docVal = cDoc.m_wordXStat.get(wid);
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDoc(k, cDoc);
					TMLikelihood += wordPerTopicLikelihood;
				}
							
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += featureLikelihood;
			}
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double>childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gamma);

		for(_ChildDoc d:pDoc.m_childDocs){
			_ChildDoc4BaseWithPhi cDoc =(_ChildDoc4BaseWithPhi)d;
			double stnLogLikelihood = 0;
			for(_Word w: stnObj.getWords()){
				int wid = w.getIndex();
			
				double wordLogLikelihood = 0;
				
				for (int k = 0; k < number_of_topics; k++) {
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDoc(k, cDoc);
					wordLogLikelihood += wordPerTopicLikelihood;
				}
				
				stnLogLikelihood += Math.log(wordLogLikelihood);
			}
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
	protected HashMap<String, Double> rankChild4StnByHybridPro(_Stn stnObj, _ParentDoc pDoc){
		HashMap<String, Double> childLikelihoodMap = new HashMap<String, Double>();
		double gammaLen = Utils.sumOfArray(m_gamma);
		
		double smoothingMu = m_LM.m_smoothingMu;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			double cDocLen = cDoc.getChildDocLenWithXVal();
						
			double stnLogLikelihood = 0;
			double alphaDoc = smoothingMu/(smoothingMu+cDocLen);
			
			for(_Word w:stnObj.getWords()){
				double featureLikelihood = 0;
				
				int wid = w.getIndex();
				
				double docVal = 0;
				if(cDoc.m_wordXStat.containsKey(wid)){
					docVal = cDoc.m_wordXStat.get(wid);
				}
				
				double LMLikelihood = (1-alphaDoc)*docVal/(cDocLen);
				
				LMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
				
				double TMLikelihood = 0;
				for(int k=0; k<number_of_topics; k++){
					double wordPerTopicLikelihood = childWordByTopicProb(k, wid)*childTopicInDoc(k, cDoc);
					TMLikelihood += wordPerTopicLikelihood;
				}
				
//				TMLikelihood += childLocalWordByTopicProb(wid, (_ChildDoc4BaseWithPhi)cDoc)*childXInDocProb(1, cDoc)/ (cDoc.getTotalDocLength() + gammaLen);
				
				featureLikelihood = m_tau*LMLikelihood+(1-m_tau)*TMLikelihood;
				featureLikelihood = Math.log(featureLikelihood);
				stnLogLikelihood += featureLikelihood;
			}
		
			double cosineSim = computeSimilarity(stnObj.m_topics, cDoc.m_xTopics[0]);
			stnLogLikelihood = m_tau*stnLogLikelihood + (1-m_tau)*cosineSim;
			
			childLikelihoodMap.put(cDoc.getName(), stnLogLikelihood);
		}
		
		return childLikelihoodMap;
	}
	
}
