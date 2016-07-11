package topicmodels;

import java.util.Arrays;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._Word;

public class LDA_GMM extends LDA_Gibbs_Debug{
	double[] m_topicProbCache_parent;
	double m_influenceLambda;
	
	public LDA_GMM(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double ksi, double tau){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, ksi, tau);
		
		m_topicProbCache_parent = new double[number_of_topics];
		Arrays.fill(m_topicProbCache_parent, 0);
		Arrays.fill(m_topicProbCache, 0);
		
//		m_influenceLambda = ;
	}
	
	
	public double calculate_E_step(_Doc d) {	
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else
			sampleInChildDoc((_ChildDoc)d);
		return 0;
	}
	
	
	protected void sampleInParentDoc(_ParentDoc d){
		int wid = 0, tid = 0;
		double normalizedProb = 0;
		
		for(_Word w:d.getWords()){
			normalizedProb = 0;
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
	
			for(tid=0; tid<number_of_topics-1; tid++){
				m_topicProbCache[tid] = wordByTopicProb(tid, wid)*topicInDocProb(tid, d);
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<m_topicProbCache[tid]; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0){
					break;
				}
			}
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	}
	
	
	protected void sampleInChildDoc(_ChildDoc d){
		
		_ParentDoc parentDoc = d.m_parentDoc;
		int wid = 0, tid = 0;
		double normalizedProb = 0;
		
		for(_Word w:d.getWords()){
			normalizedProb = 0;
			wid = w.getIndex();
			tid = w.getTopic();
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
//			for(tid=0; tid<number_of_topics; tid++){
// 				m_topicProbCache_parent[tid] = topicInDocProb(tid, parentDoc);
//			}
//			
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] = wordByTopicProb(tid, wid)*topicInDocProb(tid, d);
				m_topicProbCache[tid] = m_topicProbCache[tid]*weightInParent(tid, wid, parentDoc);
//				m_topicProbCache[tid] = Math.exp(m_topicProbCache[tid]);
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics-1; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb <= 0){
					break;
				}
			}
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
	
	}
	
	protected double topicInDocProb(int tid, _Doc d){
		return d.m_sstat[tid]/(d.getTotalDocLength()+number_of_topics*d_alpha);
	}
	
	protected double weightInParent(int tid, int wid, _Doc d){
		int topicNum = 0;
	
		for(_Word w:d.getWords()){
			if(w.getIndex() == wid)
				if(w.getTopic() == tid)
					topicNum += 1;
		}
		
		return Math.log(topicNum+1)+1;
	}
}
