package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures._APPQuery;
import structures._ChildDoc;
import structures._ChildDoc4APP;
import structures._ChildDoc4BaseWithPhi;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._ParentDoc4APP;
import structures._SparseFeature;
import structures._Word;
import utils.Utils;

public class APPACCTM_C extends ParentChildBaseWithPhi_Gibbs{
	ArrayList<_APPQuery> m_APPQueries;
	
	public APPACCTM_C(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double ksi, double tau, ArrayList<_APPQuery>appQueries){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, ksi, tau);
	
		m_APPQueries = appQueries;
	}
	

	public String toString(){
		return String.format("APP Parent Child Base Phi^c topic model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]", 
				number_of_topics, d_alpha, d_beta, m_gamma[0], m_gamma[1]);
	}
	
	protected void initialize_probability(Collection<_Doc> collection){
		createSpace();
		
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		
		for(_Doc d:collection){
			if(d instanceof _ParentDoc){
				((_ParentDoc4APP)d).setTopics4Gibbs(number_of_topics, 0);
			
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				((_ChildDoc4BaseWithPhi) d).createXSpace(number_of_topics, m_gamma.length, vocabulary_size, d_beta);
				((_ChildDoc4BaseWithPhi) d).setTopics4Gibbs(number_of_topics, 0);
				computeMu4Doc((_ChildDoc) d);
			}
			
			if(d instanceof _ParentDoc){
				for (_Word w:d.getWords()) {
					word_topic_sstat[w.getTopic()][w.getIndex()]++;
					m_sstat[w.getTopic()]++;
				}	
			}else if(d instanceof _ChildDoc4BaseWithPhi){
				for(_Word w: d.getWords()){
					int xid = w.getX();
					int tid = w.getTopic();
					int wid = w.getIndex();
					//update global
					if(xid==0){
						word_topic_sstat[tid][wid] ++;
						m_sstat[tid] ++;
					}
				}
			}
		}
		
		imposePrior();	
		m_statisticsNormalized = false;

	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		_ChildDoc4BaseWithPhi cDoc = (_ChildDoc4BaseWithPhi)d;
		int wid, tid, xid;
		double normalizedProb;
		
		for(_Word w:cDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			xid = w.getX();
			
			if(xid==0){
				cDoc.m_xTopicSstat[xid][tid] --;
				cDoc.m_xSstat[xid] --;
				cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid)-1);

				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] --;
					m_sstat[tid] --;
				}
			}else if(xid==1){
				cDoc.m_xTopicSstat[xid][wid]--;
				cDoc.m_xSstat[xid] --;
				cDoc.m_childWordSstat --;
			}
			
			normalizedProb = 0;
			double pLambdaZero = childXInDocProb(0, cDoc);
			double pLambdaOne = childXInDocProb(1, cDoc);
			
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = childWordByTopicProb(tid, wid);
				double pTopic = childTopicInDocProb(tid, cDoc);
				
				m_topicProbCache[tid] = pWordTopic*pTopic*pLambdaZero;
				normalizedProb += m_topicProbCache[tid];
			}
			
			double pWordTopic = childLocalWordByTopicProb(wid, cDoc);
			m_topicProbCache[tid] = pWordTopic*pLambdaOne;
			normalizedProb += m_topicProbCache[tid];
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache.length; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==m_topicProbCache.length)
				tid --;
			
			if(tid<number_of_topics){
				xid = 0;
				w.setX(xid);
				w.setTopic(tid);
				cDoc.m_xTopicSstat[xid][tid]++;
				cDoc.m_xSstat[xid]++;
				
				if (cDoc.m_wordXStat.containsKey(wid)) {
					cDoc.m_wordXStat.put(wid, cDoc.m_wordXStat.get(wid) + 1);
				} else {
					cDoc.m_wordXStat.put(wid, 1);
				}
				
				if(m_collectCorpusStats){
					word_topic_sstat[tid][wid] ++;
					m_sstat[tid] ++;
 				}
				
			}else if(tid==(number_of_topics)){
				xid = 1;
				w.setX(xid);
				w.setTopic(tid);
				cDoc.m_xTopicSstat[xid][wid]++;
				cDoc.m_xSstat[xid]++;
				cDoc.m_childWordSstat ++;
				
			}
		}
	}
	
	
	protected void estThetaInDoc(_Doc d){
		if(d instanceof _ParentDoc){
			Utils.L1Normalization(d.m_topics);
		}else if(d instanceof _ChildDoc){
			((_ChildDoc4BaseWithPhi)d).estGlobalLocalTheta();
		}
		m_statisticsNormalized = true;
	}
	
	public void debugOutput(String filePrefix){
		m_LM.generateReferenceModelWithXVal();
		printTopAPP4Query(filePrefix);
	}
	
	protected void printTopAPP4Query(String filePrefix){
		String topAPP4QueryFile = filePrefix + "/topAPP4Query.txt";
		
		try{
			PrintWriter pw = new PrintWriter(new File(topAPP4QueryFile));
			
			for(_APPQuery appQuery:m_APPQueries){
				pw.print(appQuery.getQueryID()+"\t");
				
				for(_Doc d:m_corpus.getCollection()){
					if(d instanceof _ParentDoc){
						_ParentDoc pDoc = (_ParentDoc)d;
						
						double likelihood = rankAPP4QueryByHybrid(appQuery, pDoc);
						pw.print(d.getTitle()+":"+likelihood);
						pw.print("\t");
					}
				}
				
				pw.println();
			}
			pw.flush();
			pw.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected double rankAPP4QueryByHybrid(_APPQuery appQuery, _ParentDoc pDoc){
		double queryLikelihood = 0.0;
		
		double smoothingMu = m_LM.m_smoothingMu;
		_SparseFeature[] pFV = pDoc.getSparse();
		
		double docLenHybridVal = 0;
		docLenHybridVal += pDoc.getDocInferLength();
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			docLenHybridVal += cDoc.m_xSstat[0];
		}
		
		double alphaDoc = smoothingMu/(smoothingMu+docLenHybridVal);
		
		for(_Word w:appQuery.getWords()){
			int wid = w.getIndex();
			double wordLoglikelihood = 0;
			double featureHybridVal = 0;
		
			int featureIndex = Utils.indexOf(pFV, wid);
			double pDocVal = 0;
			if(featureIndex != -1){
				pDocVal = pFV[featureIndex].getValue();
			}
			featureHybridVal += pDocVal;
			
			for(_ChildDoc cDoc:pDoc.m_childDocs){
				_ChildDoc4APP cDoc4APP = (_ChildDoc4APP)cDoc;
				if (cDoc4APP.m_wordXStat.containsKey(wid)) {
					featureHybridVal += cDoc4APP.m_wordXStat.get(wid);
				}
			}
		
			double wordLMLikelihood = (1-alphaDoc)*(featureHybridVal/docLenHybridVal);
			
			wordLMLikelihood += alphaDoc*m_LM.getReferenceProb(wid);
			
			double wordTMLikelihood = 0;
			for(int k=0; k<number_of_topics; k++){
				double wordPerTopicLikelihood = parentWordByTopicProb(k, wid)*topicInHybridDocProb(k, pDoc);
				wordTMLikelihood += wordPerTopicLikelihood;
			}
			
			wordLoglikelihood = (m_tau)*wordLMLikelihood+(1-m_tau)*wordTMLikelihood;
			
			queryLikelihood += Math.log(wordLoglikelihood);
		}
		
		return queryLikelihood;
	}
	
protected double topicInHybridDocProb(int tid, _ParentDoc pDoc){
		
		double prob = 0;
		
		double topicNum = 0;
		double wordNum = 0;
		double mu = 0;
		double childNum = 0;
		for(_ChildDoc cDoc:pDoc.m_childDocs){
			topicNum += cDoc.m_xTopicSstat[0][tid];
			wordNum += cDoc.m_xSstat[0];
			mu += cDoc.getMu();
			childNum += 1;
		}
		mu /= childNum;
		
		topicNum += pDoc.m_sstat[tid];
		wordNum += pDoc.getDocInferLength();		
		
		double parentInfluence = (pDoc.m_sstat[tid])/(pDoc.getDocInferLength());
		prob = (d_alpha+topicNum+mu*parentInfluence+d_alpha);
		
		prob /= (number_of_topics*d_alpha+wordNum+mu+number_of_topics*d_alpha);
		
		return prob;
		
	}
	
}
