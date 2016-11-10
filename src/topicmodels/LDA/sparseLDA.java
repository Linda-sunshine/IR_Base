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
	
	public String toString() {
		return String.format("sparseLDA[k:%d, alpha:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta, 1-m_testWord4PerplexityProportion);
	}
	
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
		
			double denominator = 0;
			denominator += DCMDoc.m_alphaDoc;
			denominator += Utils.sumOfArray(DCMDoc.m_sstat);
			
			for(tid=0; tid<number_of_topics; tid++){
				m_topicProbCache[tid] = 0;
				if(DCMDoc.m_topicIndicator[tid]==false)
					continue;
				double term1 = 0;
				term1 = topicInDocProb(tid, denominator, DCMDoc);
				term1 = wordTopicProb(tid, wid, DCMDoc);

				m_topicProbCache[tid] = topicInDocProb(tid, denominator, DCMDoc)
						* wordTopicProb(tid, wid, DCMDoc);
				p += m_topicProbCache[tid];
			}
			
			p *= m_rand.nextDouble();
			tid = -1;
			while(p>0 && tid<number_of_topics-1){
				tid ++;
				p -= m_topicProbCache[tid];
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
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat --;
				DCMDoc.m_alphaDoc -= d_alpha;
			}
			
			if(DCMDoc.m_sstat[k]>0){
				xk = true;
			}else{
				double prob = 0;
				
				double trueProb = 0;
				double falseProb = 0;
				double term1 = DCMDoc.m_alphaDoc;
				double term2 = d_alpha;
				double term3 = m_s + DCMDoc.m_indicatorTrue_stat;
				double term4 = m_t + number_of_topics-1
						- DCMDoc.m_indicatorTrue_stat;
				//double term1 = DCMDoc.m_alphaDoc+m_alpha[k], DCMDoc.m_alphaDoc+m_alpha[k]+DCMDoc.getTotalDocLength());
				//double term2 = (m_s+DCMDoc.m_indicatorTrue_stat);
				double Q = term3 / term4;
				for (int i = 0; i < DCMDoc.getTotalDocLength(); i++) {
					double QTemp = (term1 + i) / (term1 + term2 + i);
					Q *= QTemp;
				}

				falseProb = 1.0/(Q+1);
				trueProb = 1-falseProb;

				prob = m_rand.nextDouble()*(trueProb+falseProb);
				if(prob<trueProb)
					xk = true;
				else
					xk = false;
				
			}
			
			DCMDoc.m_topicIndicator[k] = xk;
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat++;	
				DCMDoc.m_alphaDoc += d_alpha;
			}
					
		}
	}
	
	protected double topicInDocProb(int tid, double denominator, _Doc4SparseDCMLDA d){
		double term1 = d.m_sstat[tid];
		term1 += d_alpha;
		
		return term1/denominator;
	}
	
	protected double wordTopicProb(int tid, int wid, _Doc d) {

		return (word_topic_sstat[tid][wid] )
				/ (m_sstat[tid] );
	}
	
	protected void collectStats(_Doc d) {
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
		
		DCMDoc.m_MStepIter += 1;
		for(int k=0; k<this.number_of_topics; k++){
			DCMDoc.m_topics[k] += DCMDoc.m_sstat[k]+d_alpha;
			if (DCMDoc.m_topicIndicator[k] == true) {
				DCMDoc.m_topicIndicator_prob[k] += 1; // miss m_s
			}
		}

		DCMDoc.m_topicIndicator_distribution += DCMDoc.m_indicatorTrue_stat;
	}
	
	protected void estThetaInDoc(_Doc d) {
		Utils.L1Normalization(d.m_topics);
		_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;

		DCMDoc.m_topicIndicator_distribution /= (DCMDoc.m_MStepIter
				* number_of_topics);
		for(int k=0; k<number_of_topics; k++)
			DCMDoc.m_topicIndicator_prob[k] /= DCMDoc.m_MStepIter;
	}
	
}
