package topicmodels;

import cern.jet.random.tdouble.Normal;
import cern.jet.random.tdouble.engine.DRand;
import structures._ChildDoc;
import structures._ChildDoc4ProbitModel;
import structures._Corpus;

public class ParentChildWithProbitModel_Gibbs extends ParentChild_Gibbs {
	public Normal m_Normal;
	public ParentChildWithProbitModel_Gibbs(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
			int number_of_topics, double alpha, double burnIn, int lag, double[] gamma, double mu){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, gamma, mu);
		m_Normal = new Normal(0, 1, new DRand());
	}
	
	public String toString(){
		return String.format("Parent Child topic model with probit model [k:%d, alpha:%.2f, beta:%.2f, gamma1:%.2f, gamma2:%.2f, Gibbs Sampling]",
				number_of_topics, d_alpha, d_beta, m_gamma[1], m_gamma[2]);
	}
	
	@Override
	void sampleInChildDoc(_ChildDoc doc){
		_ChildDoc4ProbitModel d = (_ChildDoc4ProbitModel)doc;
		
		int wid, tid, xid;
		double xVal = 0.0;
		double normalizedProb;
		
		//sampling the indicator variable 
		for(int i=0; i<d.m_words.length; i++){
			xVal = d.m_xIndicator[i];
			tid = d.m_topicAssignment[i];
					
			if(xVal > 0)
				xid = 1;	
			else
				xid = 0;
			
			d.m_xSstat[xid] --;
			d.m_xTopicSstat[xid][tid] --;
			
			xVal = xPredictiveProb(d, i);
			d.m_xIndicator[i] = xVal;
			
			if(xVal > 0)
				xid = 1;			
			else 
				xid = 0;
			
			d.m_xSstat[xid] ++;
			d.m_xTopicSstat[xid][tid] ++;			
		}
		
		for(int i=0; i<d.m_words.length; i++){
			wid = d.m_words[i];
			tid = d.m_topicAssignment[i];
			xVal = d.m_xIndicator[i];
			
			if(xVal > 0)
				xid = 1;
			else
				xid = 0;
			
			d.m_xTopicSstat[xid][tid] --;
			d.m_xSstat[xid] --;
			
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0.0;
			
			for(tid=0; tid<number_of_topics; tid++){				
				double pWordTopic = childWordByTopicProb(tid, wid);
				if(xid == 1){//p(z=tid,x=1) from specific					
					double pTopicLocal = childTopicInDocProb(tid, 1, d);
					m_topicProbCache[tid] = pWordTopic*pTopicLocal;					
				} else {//p(z=tid,x=0) from background					
					double pTopicGlobal = childTopicInDocProb(tid, 0, d);	
					m_topicProbCache[tid] = pWordTopic*pTopicGlobal;					
				}
				
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();			
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}

			if (tid == number_of_topics)
				tid--;
			
			d.m_topicAssignment[i] = tid;
		
			d.m_xTopicSstat[xid][tid] ++;
			d.m_xSstat[xid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid]++;
				m_sstat[tid]++;
			}			
		}
	}
	
	public double xPredictiveProb(_ChildDoc4ProbitModel d, int i){
		double mu = 0.0;
		double sigma = 0.0;
		
		int wid = d.m_words[i];
		double[] muVct = d.m_fixedMuPartMap.get(wid);
		for(int n=0; n<d.m_words.length; n++){
			if(n==i)
				continue;
			if(n<i)
				mu += d.m_xIndicator[n]*muVct[n];
			else if(n>i)
				mu += d.m_xIndicator[n]*muVct[n-1];
		}
		
		sigma = d.m_fixedSigmaPartMap.get(wid);
		return m_Normal.nextDouble(mu, sigma);//mean and standard deviation		
	}
}
