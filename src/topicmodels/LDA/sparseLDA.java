package topicmodels.LDA;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4SparseDCMLDA;
import structures._Word;
import utils.Utils;

public class sparseLDA extends LDA_Gibbs {
	public double m_t;
	public double m_s;
	
	public sparseLDA(int number_of_iteration, double converge, double beta,
			_Corpus c, double lambda, 
			int number_of_topics, double alpha, double burnIn, int lag, double tParam, double sParam) {
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
				alpha, burnIn, lag);

		m_t = tParam;
		m_s = sParam;
	}
	
	@Override
	public String toString() {
		return String.format("sparseLDA[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		for(int i=0; i<number_of_topics; i++)
			Arrays.fill(word_topic_sstat[i], d_beta);
		Arrays.fill(m_sstat, d_beta*vocabulary_size);
		Arrays.fill(m_topicProbCache, 0);
		
		// initialize topic-word allocation, p(w|z)
		for(_Doc d:collection) {
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;

			DCMDoc.setTopics4Gibbs(number_of_topics, d_alpha);//allocate memory and randomize it
			for(_Word w:d.getWords()) {
				word_topic_sstat[w.getTopic()][w.getIndex()] ++;
				m_sstat[w.getTopic()] ++;
			}
		}
		
		imposePrior();		
	}
	
	@Override
	public double calculate_E_step(_Doc d){
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;

		DCMDoc.permutation();
		
		sampleTopicAssignment(DCMDoc);
		sampleOnOffIndicator(DCMDoc);
		
		return 0;
	}
	
	protected void sampleTopicAssignment(_Doc4SparseDCMLDA DCMDoc){
		int wid, tid;
		double p;
		
		for(_Word w:DCMDoc.getWords()){
			wid = w.getIndex();
			tid = w.getTopic();
			
			DCMDoc.m_sstat[tid] --;
			
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] --;
				m_sstat[tid] --;
			}
			
			p = 0;
			double denominator = DCMDoc.m_alphaDoc + Utils.sumOfArray(DCMDoc.m_sstat);//why do we need this? isn't it a constant?
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] = 0;
				if(DCMDoc.m_topicIndicator[tid]==false)
					continue;				

				m_topicProbCache[tid] = topicInDocProb(tid, denominator, DCMDoc) * wordByTopicProb(tid, wid);
				p += m_topicProbCache[tid];
			}
			
			p *= m_rand.nextDouble();
			tid = 0;
			while(p>0 && tid<number_of_topics-1){
				p -= m_topicProbCache[tid];
				tid ++;
			}
			
			w.setTopic(tid);
			DCMDoc.m_sstat[tid] ++;
			
			if(m_collectCorpusStats){
				word_topic_sstat[tid][wid] ++;
				m_sstat[tid] ++;
			}
		}
		
	}
	
	protected void sampleOnOffIndicator(_Doc4SparseDCMLDA DCMDoc){
		for(int k=0; k<number_of_topics; k++){
			boolean xk = DCMDoc.m_topicIndicator[k];
			if(xk){
				DCMDoc.m_indicatorTrue_stat --;
				DCMDoc.m_alphaDoc -= d_alpha;
			}
			
			if(DCMDoc.m_sstat[k]>0){
				xk = true;
			}else{
				
				//none of them has anything to do with topic k?
				double term1 = DCMDoc.m_alphaDoc;
				double term2 = d_alpha;
				double term3 = m_s + DCMDoc.m_indicatorTrue_stat;
				double term4 = m_t + number_of_topics - 1 - DCMDoc.m_indicatorTrue_stat;
				//double term1 = DCMDoc.m_alphaDoc+m_alpha[k], DCMDoc.m_alphaDoc+m_alpha[k]+DCMDoc.getTotalDocLength());
				//double term2 = (m_s+DCMDoc.m_indicatorTrue_stat);
				double Q = term3 / term4;
				for (int i = 0; i < DCMDoc.getTotalDocLength(); i++) 
					Q *= (term1 + i) / (term1 + term2 + i);

				if(m_rand.nextDouble()<1.0/(Q+1))
					xk = false;
				else
					xk = true;				
			}
			
			DCMDoc.m_topicIndicator[k] = xk;
			if(xk){
				DCMDoc.m_indicatorTrue_stat++;	
				DCMDoc.m_alphaDoc += d_alpha;
			}					
		}
	}
	
	protected double topicInDocProb(int tid, double denominator, _Doc4SparseDCMLDA d){
		return (d.m_sstat[tid] + d_alpha)/denominator;
	}
	
	@Override
	protected void collectStats(_Doc d) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		
		DCMDoc.m_MStepIter += 1;
		for(int k=0; k<this.number_of_topics; k++){
			DCMDoc.m_topics[k] += DCMDoc.m_sstat[k]+d_alpha;
			if (DCMDoc.m_topicIndicator[k] == true) 
				DCMDoc.m_topicIndicator_prob[k] += 1; // miss m_s
		}

		DCMDoc.m_topicIndicator_distribution += DCMDoc.m_indicatorTrue_stat;
	}
	
	@Override
	protected void estThetaInDoc(_Doc d) {
		Utils.L1Normalization(d.m_topics);
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

		DCMDoc.m_topicIndicator_distribution /= (DCMDoc.m_MStepIter * number_of_topics);
		for(int k=0; k<number_of_topics; k++)
			DCMDoc.m_topicIndicator_prob[k] /= DCMDoc.m_MStepIter;
	}
	
}
