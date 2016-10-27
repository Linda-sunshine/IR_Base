package topicmodels.DCM;

import java.util.Arrays;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._Doc4SparseDCMLDA;
import structures._Word;
import utils.Utils;

public class sparseDCMLDA extends DCMLDA{
	
	public double m_t, m_s;
	
	public sparseDCMLDA(int number_of_iteration, double converge, double beta, _Corpus c, 
			double lambda, int number_of_topics, double alpha, double burnIn, int lag, int newtonIter, double newtonConverge, double tParam, double sParam){
		super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, burnIn, lag, newtonIter, newtonConverge);
		
		m_t = tParam;
		m_s = sParam;
		
		m_corpusSize = c.getSize();
		m_newtonIter = newtonIter;
		m_newtonConverge = newtonConverge;
	}
	
	protected void initialize_probability(Collection<_Doc> collection) {

		m_alpha = new double[number_of_topics];
		m_beta = new double[number_of_topics][vocabulary_size];

		m_totalAlpha = 0;
		m_totalBeta = new double[number_of_topics];

		m_alphaAuxilary = new double[number_of_topics];
		
		initialAlphaBeta();

		for (_Doc d : collection) {
			((_Doc4SparseDCMLDA)d).setTopics4Gibbs(number_of_topics, m_alpha, vocabulary_size);
		}

		imposePrior();
	}

	
	protected void initialAlphaBeta() {

		Arrays.fill(m_sstat, 0);
		Arrays.fill(m_alphaAuxilary, 0);
		for (int k = 0; k < number_of_topics; k++) {
			Arrays.fill(topic_term_probabilty[k], 0);
		}

		 for (int k = 0; k < number_of_topics; k++){
			m_alpha[k] = m_rand.nextDouble()+d_alpha;
			for (int v = 0; v < vocabulary_size; v++) {
				m_beta[k][v] = m_rand.nextDouble() + d_beta;
			}
		}
		
		m_totalAlpha = Utils.sumOfArray(m_alpha);
		for (int k = 0; k < number_of_topics; k++) {
			m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);
		}

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
			DCMDoc.m_wordTopic_stat[tid][wid]--;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid] --;
			
			p = 0;
			
//			double totalAlphaDoc = 0;
//			for(int i=0; i<number_of_topics; i++){
//				if(DCMDoc.m_topicIndicator[i]==true)
//					totalAlphaDoc += m_alpha[i];
//			}
			
			
			double denominator = 0;
			denominator += DCMDoc.m_alphaDoc;
			denominator += Utils.sumOfArray(DCMDoc.m_sstat);
			
			for(tid=0; tid<number_of_topics; tid++){
				if(DCMDoc.m_topicIndicator[tid]==false)
					continue;
				double term1 = 0;
				term1 = topicInDocProb(tid, denominator, DCMDoc);
				term1 = wordTopicProb(tid, wid, DCMDoc);
				
				m_topicProbCache[tid] = topicInDocProb(tid, DCMDoc)*wordTopicProb(tid, wid, DCMDoc);
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
			DCMDoc.m_wordTopic_stat[tid][wid] ++;
			
			if(m_collectCorpusStats)
				word_topic_sstat[tid][wid] ++;
		}
		
	}
	
	protected void sampleOnOffIndicator(_Doc4SparseDCMLDA DCMDoc){
		for(int k=0; k<number_of_topics; k++){
			boolean xk = DCMDoc.m_topicIndicator[k];
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat --;
				DCMDoc.m_alphaDoc -= m_alpha[k];
			}
			
			if(DCMDoc.m_sstat[k]>0){
				xk = true;
				
			}else{
				double prob = 0;
				
				double trueProb = 0;
				double term1 = gammaRatio(DCMDoc.m_alphaDoc+m_alpha[k], DCMDoc.m_alphaDoc+m_alpha[k]+DCMDoc.getTotalDocLength());
				double term2 = (m_s+DCMDoc.m_indicatorTrue_stat);
				trueProb = term1*term2;
				
				double falseProb = 0;
				term1 = gammaRatio(DCMDoc.m_alphaDoc, DCMDoc.m_alphaDoc+DCMDoc.getTotalDocLength());
				term2 = (m_t+number_of_topics-DCMDoc.m_indicatorTrue_stat);
				falseProb = term1*term2;
				
				prob = m_rand.nextDouble()*(trueProb+falseProb);
				if(prob<trueProb)
					xk = true;
				else
					xk = false;
				
			}
			
			DCMDoc.m_topicIndicator[k] = xk;
			if(xk==true){
				DCMDoc.m_indicatorTrue_stat++;	
				DCMDoc.m_alphaDoc += m_alpha[k];
			}
					
		}
	}
	
	protected double gammaRatio(double nominator, double denominator){
		double ratio = 0;
		
		double gap = denominator-nominator;
		for(int i=0; i<gap; i++){
			ratio *= (denominator-1-i);
		}
		
		ratio = 1.0/ratio;
		return ratio;
	}
	
	protected double topicInDocProb(int tid, double denominator, _Doc4SparseDCMLDA d){
		double term1 = d.m_sstat[tid];
		term1 += m_alpha[tid];
		
		return term1/denominator;
	}
	
	protected void updateAlpha(){
		double diff = 0;
		double smallAlpha = 0.1;
		
		int iteration = 0;
		do {

			diff = 0;
			double[] wordNum4Tid = new double[number_of_topics];

			double totalAlphaDenominator = 0;
			
			double deltaAlpha = 0;

			for (int k = 0; k < number_of_topics; k++) {
				wordNum4Tid[k] = 0;
				double totalAlphaNumerator = 0;
				totalAlphaDenominator = 0;
				for (_Doc d : m_trainSet) {
					_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
					if(DCMDoc.m_topicIndicator[k]==false)
						continue;
					
					wordNum4Tid[k] += DCMDoc.m_sstat[k];

					totalAlphaDenominator += Utils.digamma(DCMDoc.getTotalDocLength()+DCMDoc.m_alphaDoc)-Utils.digamma(DCMDoc.m_alphaDoc);
					totalAlphaNumerator += Utils.digamma(m_alpha[k]
							+ d.m_sstat[k])
							- Utils.digamma(m_alpha[k]);
				}
				
				if(wordNum4Tid[k]==0){
					deltaAlpha = 0;
				}else{
					deltaAlpha = totalAlphaNumerator*1.0/totalAlphaDenominator;
				}

				double newAlpha = m_alpha[k] * deltaAlpha+d_alpha;
				double t_diff = Math.abs(m_alpha[k] - newAlpha);
				if (t_diff > diff)
					diff = t_diff;

				m_alpha[k] = newAlpha;
			}
			
			iteration++;
	
			if(iteration > m_newtonIter)
				break;
			
		}while(diff>m_newtonConverge);

//		System.out.println("iteration\t" + iteration);
		m_totalAlpha = 0;
		for (int k = 0; k < number_of_topics; k++) {
			m_totalAlpha += m_alpha[k];
		}
	}
	
	protected void finalEst(){
		double statisticsIter = 0;
//		double term2 = 0;
		for (int j = 0; j < number_of_iteration; j++) {
			init();
			for (_Doc d : m_trainSet){
				calculate_E_step(d);

				_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
				if (j % 20 == 0) {
					statisticsIter += 1;
					for(int k=0; k<number_of_topics; k++)
						if(DCMDoc.m_topicIndicator[k]==true){
							DCMDoc.m_topicIndicator_prob[k] += 1; // miss m_s
						}

					DCMDoc.m_topicIndicator_distribution += DCMDoc.m_indicatorTrue_stat;
				}
			}
				
		}
		
		for(int k=0; k<number_of_topics; k++)
			Arrays.fill(topic_term_probabilty[k], 0);
		
		for(int k=0; k<number_of_topics; k++)
			for(int v=0; v<vocabulary_size; v++)
				topic_term_probabilty[k][v] += word_topic_sstat[k][v]+m_beta[k][v];

		
		for(int i=0; i<number_of_topics; i++)
			Utils.L1Normalization(topic_term_probabilty[i]);
		
		for(_Doc d:m_trainSet){
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA)d;
			Arrays.fill(DCMDoc.m_topics, 0);
			
			for (int k = 0; k < this.number_of_topics; k++) {
				if(DCMDoc.m_topicIndicator[k]==false)
					continue;
				DCMDoc.m_topics[k] += DCMDoc.m_sstat[k] + m_alpha[k];
	
				for (int v = 0; v < vocabulary_size; v++){
					DCMDoc.m_wordTopic_prob[k][v] += DCMDoc.m_wordTopic_stat[k][v]+m_beta[k][v];
				}
			}
		}
		
		for (_Doc d : m_trainSet) {
			estThetaInDoc(d);
			_Doc4SparseDCMLDA DCMDoc = (_Doc4SparseDCMLDA) d;
			DCMDoc.m_topicIndicator_distribution /= statisticsIter
					* number_of_topics;
			for(int k=0; k<number_of_topics; k++)
				DCMDoc.m_topicIndicator_prob[k] /= statisticsIter;
		}
	}
}
