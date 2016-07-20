package topicmodels;

import structures._ChildDoc;
import structures._Corpus;
import structures._Doc;
import structures._ParentDoc;
import structures._Word;

public class DCMCorrLDA extends DCMLDA{
	
	protected double m_alpha_c;
	
	public DCMCorrLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, int number_of_topics, 
			double alpha_a, double alpha_c, double burnIn, int lag, int newtonIter, double newtonConverge){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a, burnIn, lag, newtonIter, newtonConverge);
		
		m_alpha_c = alpha_c;
	}
	
	public String toString(){
		return String.format("DCMCorrLDA[k:%d, alpha^a:%.2f, alpha^c:%.2f, beta:%.2f, training proportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, m_alpha_c, d_beta, m_testWord4PerplexityProportion);
	}
	
	public double calculate_E_step(_Doc d){
		d.permutation();
		
		if(d instanceof _ParentDoc)
			sampleInParentDoc((_ParentDoc)d);
		else if(d instanceof _ChildDoc)
			sampleInChildDoc((_ChildDoc)d);
		
		return 0;
	}
	
	protected void sampleInParentDoc(_Doc d){
		int wid, tid;
		double normalizedProb;
		
		for(_Word w:d.getWords()){
			tid = w.getTopic();
			wid = w.getIndex();
			
			d.m_sstat[tid] --;
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			normalizedProb = 0;
			for(tid=0; tid<number_of_topics; tid++){
				double pWordTopic = parentWordByTopicProb(tid, wid);
				double pTopicPDoc = parentTopicInDocProb(tid, d);
				double pTopicCDoc = parentChildInfluenceProb(tid, d);
				
				m_topicProbCache[tid] = pWordTopic*pTopicPDoc*pTopicCDoc;
				normalizedProb += m_topicProbCache[tid];
			}
			
			normalizedProb *= m_rand.nextDouble();
			for(tid=0; tid<number_of_topics; tid++){
				normalizedProb -= m_topicProbCache[tid];
				if(normalizedProb<=0)
					break;
			}
			
			if(tid==number_of_topics)
				tid --;
			
			w.setTopic(tid);
			d.m_sstat[tid] ++;
			d.m_wor
		}
		
	}
	
	protected void sampleInChildDoc(_ChildDoc d){
		int wid, tid;
		double normalizedProb;
		
		for(_Word w:d.getWords()){
			
		}
	}

	
}

